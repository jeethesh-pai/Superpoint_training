import os

import torch
from Data_loader import TLSScanData
import yaml
from HPatches_dataset import HPatches
from torchvision import transforms
import matplotlib.pyplot as plt

config_file_path = 'HPatches_config.yaml'
with open(config_file_path) as path:
    config = yaml.full_load(path)
train_set = HPatches(transform=None, **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
for i, sample in enumerate(train_loader):
    image = sample['image'].numpy().squeeze()
    fig, axes = plt.subplots(1, 2)
    axes[1].imshow(image[2, ...].squeeze(), cmap='gray')
    axes[0].imshow(image[1, ...].squeeze(), cmap='gray')
    plt.show()
    if i == 5:
        break
