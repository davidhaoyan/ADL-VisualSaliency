import torch
import numpy as np
from train import MrCNN
from torch import nn
from torch.utils.data import DataLoader
from dataset import MIT
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.io import read_file, decode_image
from torchvision import transforms
from metrics import calculate_auc, roc_auc
import os
import argparse
from datetime import datetime
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(
    description="Inference MrCNN",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint-file", 
    default="./good_ones/checkpoint_epoch_21_2024-11-21_16-08-42.pth.tar",
    type=str,
)
parser.add_argument(
    "--threshold",
    default=0.5,
    type=float,
)
parser.add_argument(
    "--output-dir",
    default="./output",
    type=str,
)

class Inferencer: 
    def __init__(
        self, 
        model: nn.Module, 
        checkpoint_file: str, 
        data_loader: DataLoader,
        device: torch.device,
        threshold: float,
    	output_dir: str,
    ): 
        self.model = model 
        self.model.eval() 
        if (checkpoint_file) != "":
            self.load_checkpoint(checkpoint_file)
        self.device = device
        self.data_loader = data_loader
        self.threshold = threshold
        self.output_dir = output_dir
        
    def load_checkpoint(self, checkpoint_file: str):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_file}")
    
    def test_model(self):
        self.model.eval()
        self.model.to(self.device)

        os.makedirs(self.output_dir, exist_ok=True)

        predictions = {}
        ground_truth = {}
        with torch.no_grad():
            for index, (batch, _, file) in enumerate(self.data_loader):
                assert file[0] == file[-1]
                file = file[0]
                file_stripped = file[:file.index('.')]
                fix_map = load_gt_fixation_map(file)
                height, width = fix_map.shape

                batch = batch.to(self.device)
                output = self.model.forward(batch)
                
                output = output.view(1, 50, 50)
                
                output_normalized = 1 / (1 + torch.exp(-output))
                output_normalized = (output_normalized - output_normalized.min()) / (output_normalized.max() - output_normalized.min())
                prediction_tensor = output_normalized
                
                rescaler = transforms.Resize(size=(height,width), interpolation=transforms.InterpolationMode.BILINEAR)
                prediction_tensor_rescaled = rescaler(prediction_tensor)

                prediction_numpy = prediction_tensor_rescaled.squeeze(0).cpu().numpy()
                predictions[file] = prediction_numpy
                ground_truth[file] = fix_map

                print(f"{file_stripped} auc: {roc_auc(prediction_numpy, fix_map)}")

                #Save image
                output_dir_raw = f"{self.output_dir}/raw"
                os.makedirs(output_dir_raw, exist_ok=True)
                save_path_raw = os.path.join(output_dir_raw, f"{file_stripped}_prediction_raw.png")
                save_image(prediction_tensor_rescaled, threshold=float('-inf'), gaussian_blur=False, save_path=save_path_raw)

                output_dir_processed = f"{self.output_dir}/processed"
                os.makedirs(output_dir_processed, exist_ok=True)
                save_path_processed = os.path.join(output_dir_processed, f"{file_stripped}_prediction.png")
                save_image(prediction_tensor_rescaled, threshold=self.threshold, gaussian_blur=True, save_path=save_path_processed)

        auc = calculate_auc(predictions, ground_truth)
        print(f"Test AUC: {auc}")

def load_gt_fixation_map(file: str) -> torch.Tensor:
    fix_map = Image.open(f"./gt/{file[:file.index('.')]}_gt.jpg")
    return np.array(fix_map)

def save_image(image: torch.Tensor, threshold: float = float('-inf'), gaussian_blur: bool = False, save_path: str = ""):
    image = torch.where(image > threshold, image, 0) #threshold=-inf -> identity function
    image = image.squeeze(0).cpu().numpy()
    if gaussian_blur:    
        image = gaussian_filter(image, 15.0)
    image = (image - np.min(image)) * 255 /(np.max(image) - np.min(image))
    image = Image.fromarray(image.astype('uint8'))
    assert(save_path != "")
    image.save(save_path)

def main(args):
    model = MrCNN(in_channels=3, class_count=1, dropout=0)

    test_dataset = MIT("./test_data.pth.tar")
    test_loader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=8)
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(DEVICE)
    
    inferencer = Inferencer(
        model=model, 
        checkpoint_file = args.checkpoint_file, 
        data_loader=test_loader,
        device = DEVICE,
        threshold = args.threshold,
    	output_dir = args.output_dir
    )
    inferencer.test_model()

if __name__ == "__main__":
    main(parser.parse_args())
    
