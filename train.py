import time
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from dataset import MIT
from metrics import calculate_auc
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import argparse
from multiprocessing import cpu_count
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter
from PIL import Image

parser = argparse.ArgumentParser(
    description="Train a MrCNN on MIT",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--learning-rate", 
    default=0.0001, 
    type=float, 
    help="Learning rate",
)
parser.add_argument(
    "--batch-size",
    default=256,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=20,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--checkpoint-file",
    default="",
    type=str,
    help="Path of checkpoint to load",
)
parser.add_argument(
    "--checkpoint-path",
    default="/user/work/lj21689/adl/cw/deep_learning/checkpoints",
    type=str,
    help="Where to save checkpoints",
)
parser.add_argument(
    "--log-dir", 
    default="./logs",
    type=str,
)
parser.add_argument(
    "--data-dir",
    default="/user/work/lj21689",
    type=str,
)
parser.add_argument(
    "--notes",
    default="",
    type=str,
)
parser.add_argument(
    "--initial-momentum",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--final-momentum",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--weight-decay",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--dropout",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--augment",
    default="off",
    type=bool,
)

EPSILON = 0.001

class MrCNN(nn.Module):
    def __init__(self, in_channels: int, class_count: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.stream = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=(7,7), stride=1, padding=0),
            #nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),

            nn.Conv2d(96, 160, kernel_size=(3,3), stride=1, padding=0),
            #nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),

            nn.Conv2d(160,288, kernel_size=(3,3), stride=1, padding=0),
            #nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            self.dropout
        )
        
        self.fc1 = nn.Linear(288*3*3, 512)
        self.fc2 = nn.Linear(288*3*3, 512)
        self.fc3 = nn.Linear(288*3*3, 512)
        
        self.fc4 = nn.Linear(512*3, class_count)
        
        self.apply(self.initialise_layer)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(1, 0, 2, 3, 4) 
        x0, x1, x2 = images[0], images[1], images[2]

        x0 = self.stream(x0)
        x1 = self.stream(x1)
        x2 = self.stream(x2)
        
        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        x0 = self.dropout(F.relu(self.fc1(x0)))
        x1 = self.dropout(F.relu(self.fc2(x1)))
        x2 = self.dropout(F.relu(self.fc3(x2)))
        #x0 = F.relu(self.fc1(x0))
        #x1 = F.relu(self.fc2(x1))
        #x2 = F.relu(self.fc3(x2))
        
        x = torch.cat((x0, x1, x2), dim=1)
        x = self.dropout(self.fc4(x))
        #x = self.fc4(x)
        return x
    
    @staticmethod
    def initialise_layer(layer):
        if isinstance(layer, nn.Conv2d):  # Initialize Conv2d layers
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):  # Initialize Linear layers
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):  # Initialize BatchNorm2d layers
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        summary_writer: SummaryWriter,
        checkpoint_path: str,
        data_dir: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.step = 0
        
    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_file = f"{self.checkpoint_path}/checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth.tar"
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {checkpoint_file}")
        return checkpoint['epoch']  # Return the starting epoch
    
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int,
        log_frequency: int,
        initial_momentum: int,
        final_momentum: int,
        checkpoint_file: str = "",
    ):
        start_epoch = 0
        if checkpoint_file != "":
            start_epoch = self.load_checkpoint(checkpoint_file)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch}:")

            current_momentum = initial_momentum + (final_momentum - initial_momentum) * (epoch / epochs)
            self.optimizer.param_groups[0]['momentum'] = current_momentum
            
            self.model.train()
            data_load_start_time = time.time()
            for (batch, labels, _) in self.train_loader:
                data_load_end_time = time.time()
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                output = self.model.forward(batch).squeeze(1)

                self.optimizer.zero_grad()
                loss = self.criterion(output, labels.float())
                loss.backward()
                self.optimizer.step()
            
                with torch.no_grad():    
                    predictions = torch.where(output > 0, 1, 0)
                    accuracy = compute_accuracy(predictions, labels)
            
                    data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)
                self.step += 1
            
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate() 

            self.save_checkpoint(epoch)
            
            
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = (self.step % len(self.train_loader)) + 1
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
        
    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )


    def validate(self):
        evaluate_model(
            data_loader=self.val_loader,
            mode="Validation",
            model=self.model,
            device=self.device,
            summary_writer=self.summary_writer,
            step=self.step,
            data_dir=self.data_dir,
        )
        
def compute_accuracy(labels, preds) -> float:
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def load_gt_fixation_map(file: str, data_dir: str) -> torch.Tensor:
    fix_map = Image.open(f"{data_dir}/ALLFIXATIONMAPS/ALLFIXATIONMAPS/{file[:file.index('.')]}_fixMap.jpg")
    return np.array(fix_map)

def evaluate_model(data_loader: DataLoader, 
                   mode: str,
                   model: nn.Module,
                   device: torch.device,
                   summary_writer: SummaryWriter,
                   step: int,
                   data_dir: str,
):
    model.eval()
    predictions = {}
    ground_truth = {}

    """if mode == "Test":
        output_dir = f"./output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(output_dir, exist_ok=True)"""
    with torch.no_grad():
        for index, (batch, _, file) in enumerate(data_loader):
            assert file[0] == file[-1]
            file = file[0]
            
            fix_map = load_gt_fixation_map(file, data_dir)
            height, width = fix_map.shape

            batch = batch.to(device)
            output = model.forward(batch)
            
            output = output.view(1, 50, 50)
            output_activation = 1 / (1 + torch.exp(-output))
            prediction_tensor = output_activation
            
            rescaler = transforms.Resize(size=(height,width), interpolation=transforms.InterpolationMode.BILINEAR)
            prediction_tensor_rescaled = rescaler(prediction_tensor)
            prediction_numpy = prediction_tensor_rescaled.squeeze(0).detach().cpu().numpy() 
            
            """if mode == "Test":
                save_path = os.path.join(output_dir, f"{file[:file.index('.')]}_prediction.png")
                prediction_numpy = (prediction - np.min(prediction)) * 255 / (np.max(prediction) - np.min(prediction))
                prediction_image = Image.fromarray(prediction_numpy.astype('uint8'))
                prediction_image.save(save_path)"""

            predictions[file] = prediction_numpy
            ground_truth[file] = fix_map

    auc = calculate_auc(predictions, ground_truth)
    print(f"{mode} evaluation AUC: {auc}")

    summary_writer.add_scalar(f"AUC/{mode}", auc, step)
        
def test_model(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    summary_writer: SummaryWriter,
    data_dir: str,
):
    evaluate_model(
        data_loader=test_loader,
        mode="Test",
        model=model,
        device=device,
        summary_writer=summary_writer,
        step=0,
        data_dir=data_dir,
    )
        
def main(args):
    data_dir = args.data_dir
    augment = False if args.augment == "off" else True
    train_dataset = MIT(f"{data_dir}/train_data.pth.tar", augment=augment)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_count, pin_memory=True) 
    
    val_dataset = MIT(f"{data_dir}/val_data.pth.tar", augment=augment)
    val_loader = DataLoader(val_dataset, batch_size=2500, shuffle=False, num_workers=args.worker_count, pin_memory=True)

    test_dataset = MIT(f"{data_dir}/test_data.pth.tar", augment=augment)
    test_loader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=args.worker_count, pin_memory=True)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    model = MrCNN(in_channels=3, class_count=1, dropout=args.dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.initial_momentum, weight_decay=args.weight_decay)
    log_dir = f"./logs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr{args.learning_rate}_epochs_{args.epochs}_notes_{args.notes}/"
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5,
    )

    checkpoint_path = f"{args.checkpoint_path}/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr{args.learning_rate}_epochs_{args.epochs}_notes_{args.notes}"
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        DEVICE,
        summary_writer,
        checkpoint_path,
        args.data_dir
    )

    final_momentum = args.initial_momentum if args.final_momentum == 0.0 else args.final_momentum
    trainer.train(
        epochs = args.epochs,
        val_frequency = args.val_frequency,
        print_frequency = args.print_frequency,
        log_frequency = args.log_frequency,
        initial_momentum = args.initial_momentum,
        final_momentum = args.final_momentum,
        checkpoint_file = args.checkpoint_file,
    )
    
    test_model(
        test_loader=test_loader,
        model=model, 
        device=DEVICE, 
        summary_writer=summary_writer,
        data_dir=args.data_dir
    )
    summary_writer.close()

    
if __name__ == "__main__":
    main(parser.parse_args())
