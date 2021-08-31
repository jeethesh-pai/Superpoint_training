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
                    default="detector_training.yaml")
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
            train_bar.set_description(f"Training Epoch -- {n_iter + 1} Loss: {running_loss / (i + 1)}, IoU: {batch_iou}")
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
            val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} Validation loss: {running_val_loss / (j + 1)}, "
                                    f"Validation IoU: {val_batch_iou}")
        running_val_loss /= len(val_loader)
        if prev_val_loss == 0:
            prev_val_loss = running_val_loss
            print('saving best model .... ')
            torch.save(Net, "saved_path/detector_training/best_model.pt")
        if prev_val_loss > running_val_loss:
            torch.save(Net, "saved_path/detector_training/best_model.pt")
            print('saving best model .... ')
            prev_val_loss = running_val_loss
        print(f"Epoch {n_iter + 1}:  val_loss: {running_val_loss}, val_IoU: {val_batch_iou}")
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
        writer.add_scalar('IoU', batch_iou, n_iter + 1)
        writer.add_scalar('Val_IoU', val_batch_iou, n_iter + 1)
        writer.flush()
        n_iter += 1
        #        (_, loss), (_, iou) = Net.train_mode(sample).items()
        #         loss.backward()
        #         if count == 0:
        #             batch_iou = iou
        #         batch_iou = torch.add(batch_iou, iou) / 2
        #         running_loss += loss.item()
        #         optimizer.step()
        #     print(f'Training Loss for the epoch-{n_iter}: {running_loss / len(train_loader)}, IoU: {batch_iou}')
        #     val_loss, val_iou = 0, 0
        #     for count_val, sample_val in tqdm.tqdm(enumerate(val_loader)):
        #         with torch.no_grad():
        #             (_, curr_loss), (_, curr_iou) = Net.train_mode(sample_val).items()
        #             val_loss += curr_loss
        #             if count_val == 0:
        #                 val_iou = curr_iou
        #             val_iou = (val_iou + curr_iou) / 2
        #     if n_iter == 0:
        #         prev_val_loss = val_loss
        #         torch.save(Net, "saved_path/best_model.pt")
        #     if val_loss < prev_val_loss and n_iter > 0:
        #         print('saving best model...')
        #         torch.save(Net, "saved_path/best_model.pt")
        #     print(f'Val Loss for the epoch-{n_iter}: {val_loss / len(val_loader)}, Val_IoU: {val_iou}')
        #     writer.add_scalar('Loss', running_loss, n_iter + 1)
        #     writer.add_scalar('Val_loss', val_loss, n_iter + 1)
        #     writer.add_scalar('IoU', batch_iou, n_iter + 1)
        #     writer.add_scalar('Val_IoU', val_iou, n_iter + 1)
        #     writer.flush()
        #     n_iter += 1
        # writer.close()
