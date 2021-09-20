import torch
import yaml
from Data_loader import TLSScanData
from Synthetic_dataset_loader import SyntheticDataset
from model_loader import SuperPointNet, load_model, SuperPointNet_gauss2
import torch.optim as optim
from utils import detector_loss, descriptor_loss_2
from torchsummary import summary
import copy
import os
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import tqdm
import warnings
import argparse
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description="This scripts helps to train Superpoint for detector training after"
                                             "Homographic adaptation labels are made as mentioned in the paper")
parser.add_argument('--config', help='Path to config file',
                    default="synthetic_shape_training.yaml")
args = parser.parse_args()
print(args)
config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
batch_size = config['model']['batch_size']
det_threshold = config['model']['detection_threshold']  # detection threshold to threshold the detector heatmap
size = config['data']['preprocessing']['resize']  # width, height
train_set = SyntheticDataset(transform=None, task='train', **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = SyntheticDataset(transform=None, task='validation', **config)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=True)
Net = SuperPointNet()
optimizer = optim.Adam(Net.parameters(), lr=config['model']['learning_rate'])
max_iter = config['train_iter']  # also called as epochs
n_iter = 0  # 1 iteration refers to the time taken to update the parameters ie. one batch = 1 iteration
# as per magicpoint paper. It was trained for 200,000 iteration i.e, 200,000 batches
# with batch size of 4 it takes 22500 iteration to complete an epoch. Therefore approximately 10 epochs are needed to
# train the magicpoint network
if torch.cuda.is_available():
    Net.cuda()
summary(Net, (1, size[1], size[0]), batch_size=1)
train_bar = tqdm.tqdm(train_loader)
prev_val_loss = 0
writer = SummaryWriter(log_dir="logs/magicpoint_training")
writer.add_graph(Net, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).cuda())
while n_iter < max_iter:
    running_loss, batch_iou = 0, 0
    Net.train(mode=True)
    for i, sample in enumerate(train_bar):
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
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss / (i + 1)},"
                                  f" IoU: {batch_iou}")
    val_bar = tqdm.tqdm(val_loader)
    Net.train(mode=False)
    running_val_loss, val_batch_iou = 0, 0
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
    if n_iter == 0:
        prev_val_loss = running_val_loss
        print('saving best model .... ')
        torch.save(copy.deepcopy(Net.state_dict()), "saved_path/magicpoint/magicpoint.pt")
    if prev_val_loss > running_val_loss:
        torch.save(copy.deepcopy(Net.state_dict()), "saved_path/magicpoint/magicpoint.pt")
        print('saving best model .... ')
        prev_val_loss = running_val_loss
    writer.add_scalar('Loss', running_loss, n_iter + 1)
    writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
    writer.add_scalar('IoU', batch_iou, n_iter + 1)
    writer.add_scalar('Val_IoU', val_batch_iou, n_iter + 1)
    writer.flush()
    n_iter += len(train_loader) * batch_size

