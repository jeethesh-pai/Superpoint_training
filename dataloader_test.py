import cv2
import numpy as np
import torch
import yaml
from utils import inv_warp_image_batch, my_inv_warp_image_batch
from model_loader import SuperPointNetBatchNorm, semi_to_heatmap
from torchsummary import summary
from HPatches_dataset import HPatches
import matplotlib.pyplot as plt

config_file_path = 'HPatches_config.yaml'
with open(config_file_path) as path:
    config = yaml.full_load(path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SuperPointNetBatchNorm()
model_weights = torch.load(config['pretrained'], map_location=device)
model.load_state_dict(model_weights)
model.to(device)
summary(model, input_size=(1, 240, 320))
train_set = HPatches(transform=None, **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
for k, sample in enumerate(train_loader):
    if sample['change'][0] == "viewpoint":
        images = sample['image']
        cropped_images = [sample['cropped_image'][i].numpy().squeeze() for i in range(len(images))]
        cropped_heatmap = [model(sample['cropped_image'][i].unsqueeze(0).to(device)) for i in range(len(images))]
        fig0, axes0 = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes0[i, j].imshow(cropped_images[count], cmap='gray')
                axes0[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        fig, axes = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes[i, j].imshow(images[count].numpy().squeeze(), cmap='gray')
                axes[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        fake_warp = [my_inv_warp_image_batch(images[0] * 255, sample['inv_homography'][0, i, ...],
                                             target_size=(images[i].shape[1], images[i].shape[2]))
                     for i in range(len(images))]
        fake_warp_cv = [cv2.warpPerspective(images[0].numpy().squeeze() * 255, sample['homography'][0, i, ...].numpy(),
                                            dsize=(images[i].shape[2], images[i].shape[1]), flags=cv2.INTER_LINEAR)
                        for i in range(len(images))]
        fig2, axes2 = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes2[i, j].imshow(fake_warp[count].numpy().squeeze(), cmap='gray')
                axes2[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        # overlaying
        new_images = [cv2.addWeighted(sample['image'][i].numpy().squeeze() * 255, 0.5, fake_warp_cv[i], 0.5, 0)
                      for i in range(len(sample['image']))]
        fig3, axes3 = plt.subplots(2, len(images) // 2)
        count = 0
        for i in range(2):
            for j in range(len(images) // 2):
                axes3[i, j].imshow(new_images[count], cmap='gray')
                axes3[i, j].set_title(f"{sample['name'][count][-5:]}")
                count += 1
        plt.show()
