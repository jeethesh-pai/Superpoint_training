import torch
import yaml
from Data_loader import TLSScanData
from model_loader import SuperPointNet, load_model
import torch.optim as optim
from utils import flattenDetection, labels2Dto3D, detector_loss, descriptor_loss_2, descriptor_loss
from torchsummary import summary
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import tqdm
import warnings
import argparse
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description="This scripts helps to train Superpoint for detector training after"
                                             "Homographic adaptation labels are made as mentioned in the paper")
parser.add_argument('--config', help='Path to config file',
                    default="joint_training.yaml")
args = parser.parse_args()

config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
batch_size = config['model']['batch_size']
numHomIter = config['data']['augmentation']['homographic']['num']
det_threshold = config['model']['detection_threshold']  # detection threshold to threshold the detector heatmap
size = config['data']['preprocessing']['resize']  # width, height
train_set = TLSScanData(transform=None, task='train', **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = TLSScanData(transform=None, task='validation', **config)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=True)
Net = SuperPointNet()
optimizer = optim.Adam(Net.parameters(), lr=config['model']['learning_rate'])
epochs = 0
Net = load_model(config['pretrained'], Net)
if torch.cuda.is_available():
    Net.cuda()
summary(Net, (1, size[1], size[0]), batch_size=1)
if config['data']['detector_training']:  # we bootstrap the Superpoint detector using homographic adapted labels
    writer = SummaryWriter(log_dir="logs/detector_training")
    writer.add_graph(Net, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).cuda())
    max_iter = config['train_iter']  # also called as epochs
    n_iter = 0
    prev_val_loss = 0
    while n_iter < max_iter:  # epochs can be lesser since no random homographic adaptation is involved
        running_loss, batch_iou = 0, 0
        train_bar = tqdm.tqdm(train_loader)
        for i, sample in enumerate(train_bar):  # make sure the homographic adaptation is false here
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(sample['image'].numpy()[0, :, :].squeeze(), cmap='gray')
            # axes[1].imshow(sample['label'][0, :, :].numpy().squeeze(), cmap='gray')
            # plt.show()
            plt.imshow(sample['label'][0, :, :].numpy().squeeze(), cmap='gray')
            plt.show()
            if torch.cuda.is_available():
                sample['image'] = sample['image'].to('cuda')
                sample['label'] = sample['label'].to('cuda')
            optimizer.zero_grad()
            out = Net(sample['image'])
            semi, _ = out['semi'], out['desc']
            det_out = detector_loss(sample['label'], semi, det_threshold=det_threshold)
            loss, iou = det_out['loss'], det_out['iou']
            if i == 0:
                batch_iou = iou
            else:
                batch_iou = (batch_iou + iou) / 2
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss / (i + 1)},"
                                      f" IoU: {batch_iou}")
        running_val_loss, val_batch_iou = 0, 0
        val_bar = tqdm.tqdm(val_loader)
        for j, val_sample in enumerate(val_bar):
            if torch.cuda.is_available():
                val_sample['image'] = val_sample['image'].to('cuda')
                val_sample['label'] = val_sample['label'].to('cuda')
            with torch.no_grad():
                val_out = Net(val_sample['image'])
                val_det_out = detector_loss(val_sample['label'], val_out['semi'], det_threshold)
                running_val_loss += val_det_out['loss'].item()
                if val_batch_iou == 0:
                    val_batch_iou = val_det_out['iou']
                else:
                    val_batch_iou = (val_batch_iou + val_det_out['iou']) / 2
            val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} / {max_iter} - Validation loss: "
                                    f"{running_val_loss / (j + 1)}, Validation IoU: {val_batch_iou}")
        running_val_loss /= len(val_loader)
        if prev_val_loss == 0:
            prev_val_loss = running_val_loss
            print('saving best model .... ')
            torch.save(Net, "saved_path/detector_training_2/best_model.pt")
        if prev_val_loss > running_val_loss:
            torch.save(Net, "saved_path/detector_training_2/best_model.pt")
            print('saving best model .... ')
            prev_val_loss = running_val_loss
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
        writer.add_scalar('IoU', batch_iou, n_iter + 1)
        writer.add_scalar('Val_IoU', val_batch_iou, n_iter + 1)
        writer.flush()
        n_iter += 1
else:  # start descriptor training with the homographically adapted model
    writer = SummaryWriter(log_dir="logs/descriptor_training")
    writer.add_graph(Net, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).cuda())
    max_iter = config['train_iter']  # also called as epochs
    det_threshold = config['model']['detection_threshold']
    n_iter = 0
    prev_val_loss = 0
    while n_iter < max_iter:  # epochs can be lesser since no random homographic adaptation is involved
        running_loss, batch_iou = 0, 0
        train_bar = tqdm.tqdm(train_loader)
        for i, sample in enumerate(train_bar):  # make sure the homographic adaptation is set to true here
            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0].imshow(sample['image'].numpy()[0, 0, :, :].squeeze(), cmap='gray')
            # axes[1, 0].imshow(sample['warped_image'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # axes[0, 1].imshow(sample['label'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # axes[1, 1].imshow(sample['warped_label'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # plt.show()
            if torch.cuda.is_available():
                sample['image'] = sample['image'].to('cuda')
                sample['label'] = sample['label'].to('cuda')
                sample['warped_image'] = sample['warped_image'].to('cuda')
                sample['warped_label'] = sample['warped_label'].to('cuda')
            optimizer.zero_grad()
            out = Net(sample['image'])
            out_warp = Net(sample['warped_image'])
            semi, desc = out['semi'], out['desc']
            semi_warped, desc_warp = out_warp['semi'], out_warp['desc']
            det_loss = detector_loss(sample['label'], semi, det_threshold=det_threshold)
            det_warp_loss = detector_loss(sample['label'], semi_warped, det_threshold)
            desc_loss = descriptor_loss_2(desc, desc_warp, homography=sample['homography'],
                                          margin_neg=config['model']['negative_margin'],
                                          margin_pos=config['model']['positive_margin'],
                                          lambda_d=config['model']['lambda_d'],
                                          threshold=config['model']['descriptor_dist'],
                                          valid_mask=sample['warped_mask'])
            total_loss = det_loss['loss'] + det_warp_loss['loss'] + config['model']['lambda_loss'] * desc_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            if i == 0:
                batch_iou = det_loss['iou'] + det_warp_loss['iou']
            else:
                batch_iou += (det_loss['iou'] + det_warp_loss['iou']) / 2
            train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss / (i + 1)},"
                                      f" IoU: {batch_iou}")
        running_val_loss, val_batch_iou = 0, 0
        val_bar = tqdm.tqdm(val_loader)
        for j, val_sample in enumerate(val_bar):
            if torch.cuda.is_available():
                val_sample['image'] = val_sample['image'].to('cuda')
                val_sample['label'] = val_sample['label'].to('cuda')
                val_sample['warped_image'] = val_sample['warped_image'].to('cuda')
                val_sample['warped_label'] = val_sample['warped_label'].to('cuda')
            with torch.no_grad():
                out = Net(val_sample['image'])
                out_warp = Net(val_sample['warped_image'])
                semi, desc = out['semi'], out['desc']
                semi_warped, desc_warp = out_warp['semi'], out_warp['desc']
                det_loss = detector_loss(val_sample['label'], semi, det_threshold=det_threshold)
                det_warp_loss = detector_loss(val_sample['label'], semi_warped, det_threshold)
                desc_loss = descriptor_loss_2(desc, desc_warp, homography=val_sample['homography'],
                                              margin_neg=config['model']['negative_margin'],
                                              margin_pos=config['model']['positive_margin'],
                                              lambda_d=config['model']['lambda_d'],
                                              threshold=config['model']['descriptor_dist'],
                                              valid_mask=val_sample['warped_mask'])
                total_loss = det_loss['loss'] + det_warp_loss['loss'] + config['model']['lambda_loss'] * desc_loss
                running_val_loss += total_loss.item()
                if j == 0:
                    val_batch_iou = det_loss['iou'] + det_warp_loss['iou']
                else:
                    val_batch_iou += (det_loss['iou'] + det_warp_loss['iou']) / 2
                val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} / {max_iter} - Validation loss: "
                                        f"{running_val_loss / (j + 1)}, Validation IoU: {val_batch_iou}")
        running_val_loss /= len(val_loader)
        if prev_val_loss == 0:
            prev_val_loss = running_val_loss
            print('saving best model .... ')
            torch.save(Net, "saved_path/joint_training/best_model.pt")
        if prev_val_loss > running_val_loss:
            torch.save(Net, "saved_path/joint_training/best_model.pt")
            print('saving best model .... ')
            prev_val_loss = running_val_loss
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
        writer.add_scalar('IoU', batch_iou, n_iter + 1)
        writer.add_scalar('Val_IoU', val_batch_iou, n_iter + 1)
        writer.flush()
        n_iter += 1
    writer.close()
