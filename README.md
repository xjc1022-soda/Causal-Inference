# Causal-Inference
```
This repository for Summer research with Dr. Luquan Yu.   
The topic is Causal Inference for treatment effect with Deep Learning Alogrithms
```
🔧 Skills Involved  

![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=linux&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Packages-Pytorch-informational?style=flat&logo=docker&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Packages-Numpy-informational?style=flat&logo=kubernetes&logoColor=white&color=2bbc8a)

## Data preprocessing

### train-valid split(5 fold crossvalidation method)
```
utils.py
```
### Mask and data augmentation
```
dataset.py
```
### Dataset and Data loader
```
datamodule.py
```

## Visualization 

### KM-Line for survival rate visualization
![High Score Group](https://github.com/xjc1022-soda/Causal-Inference/blob/main/km.jpg)
```
Python km_lifeline.py
```

### CT images sample
![CT](https://github.com/xjc1022-soda/Causal-Inference/blob/main/simclr.jpg)

### Training results demo in Wandb
![Train](https://github.com/xjc1022-soda/Causal-Inference/blob/main/train_loss.png)  

## Models 
To solve the problem of limited CT images
```
survival_prediction.py 👉 pre-trained models   
simclr.py 👉 semi-supervised models
```

## Temperary report
👀 [Research Report](https://github.com/xjc1022-soda/xjc1022-soda/blob/main/summer_research_proposal.pdf)  
