import os

import torch
from Data_loader import TLSScanData
import yaml
from utils import inv_warp_image_batch
from HPatches_dataset import HPatches
from torchvision import transforms
import matplotlib.pyplot as plt

config_file_path = 'HPatches_config.yaml'
with open(config_file_path) as path:
    config = yaml.full_load(path)
train_set = HPatches(transform=None, **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
for k, sample in enumerate(train_loader):
    if sample['change'][0] == "Viewpoint":
        images = sample['image'].squeeze()
        fig, axes = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes[i, j].imshow(images[count, ...])
                axes[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        images = torch.cat([sample['image'][0, 0, ...]]*(sample['image'].shape[1]), dim=0)
        fake_warp = inv_warp_image_batch(images.squeeze() * 255, sample['inv_homography'].squeeze())
        fig2, axes2 = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes2[i, j].imshow(fake_warp[count, ...])
                axes2[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        plt.show()
