# ADL-VisualSaliency
Dataset: MIT: https://people.csail.mit.edu/tjudd/WherePeopleLook/

# Training
Ground truth fixation maps can be downloaded from the link above  
Trained on BC4  
```python train.py --epochs 200 --data-dir {} --checkpoint-path {where to save checkpoints} --learning-rate 0.0001 --batch-size 512```  
--augment "on"/"off" for data augmentation  
Replace ```train.py``` with ```train_bn.py``` for batch normalization  

# Inference
Ground truth fixation maps are loaded from ./gt
```python inference.py --checkpoint-file {path to checkpoint to load} --output-dir {where to save output}```  
Replace ```inference.py``` with ```inference_bn.py``` 
