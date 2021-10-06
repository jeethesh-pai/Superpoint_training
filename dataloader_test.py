import os

import torch
from Data_loader import TLSScanData
import yaml
from torchvision import transforms
import matplotlib.pyplot as plt

config_file_path = 'tls_scan_superpoint_config_file.yaml'
with open(config_file_path) as path:
    config = yaml.load(path)
train_set = TLSScanData(transform=None, task='val', **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)
for i, sample in enumerate(train_loader):
    image = sample['image'].numpy().squeeze()
    label = sample['warped_image'].numpy().squeeze()
    fig, axes = plt.subplots(1, 2)
    axes[1].imshow(label[1, 0, ...], cmap='gray')
    axes[0].imshow(image[1, ...], cmap='gray')
    plt.show()
    if i == 5:
        break
