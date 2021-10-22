import torch
import yaml
from Data_loader import TLSScanData
from Synthetic_dataset_loader import SyntheticDataset
from model_loader import SuperPointNetBatchNorm2, ModelWrapper
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
Net = SuperPointNetBatchNorm2()
optimizer = optim.Adam(Net.parameters(), lr=config['model']['learning_rate'])
max_iter = config['train_iter']  # also called as epochs
n_iter = 0  # 1 iteration refers to the time taken to update the parameters ie. one batch = 1 iteration
# as per magicpoint paper. It was trained for 200,000 iteration i.e, 200,000 batches
# with batch size of 4 it takes 22500 iteration to complete an epoch. Therefore approximately 10 epochs are needed to
# train the magicpoint network
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net.to(device)
summary(Net, (1, size[1], size[0]), batch_size=1)
wrappedModel = ModelWrapper(Net)
prev_val_loss = None
writer = SummaryWriter(log_dir="logs/magicpoint_training")
writer.add_graph(wrappedModel, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).to(device))
while n_iter < max_iter:
    running_loss = 0
    Net.train(mode=True)
    train_bar = tqdm.tqdm(train_loader)
    for i, sample in enumerate(train_bar):
        sample['image'] = sample['image'].to(device)
        sample['label'] = sample['label'].to(device)
        optimizer.zero_grad()
        out = Net(sample['image'])
        semi, _ = out['semi'], out['desc']
        det_out = detector_loss(sample['label'], semi, device=device)
        loss = det_out['loss']
        loss.backward()
        optimizer.step()
        running_loss = (running_loss + loss.item()) / 2
        train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss}")
    val_bar = tqdm.tqdm(val_loader)
    Net.train(mode=False)
    running_val_loss = 0
    for j, val_sample in enumerate(val_bar):
        val_sample['image'] = val_sample['image'].to(device)
        val_sample['label'] = val_sample['label'].to(device)
        with torch.no_grad():
            val_out = Net(val_sample['image'])
            val_det_out = detector_loss(val_sample['label'], val_out['semi'], device)
            running_val_loss += val_det_out['loss'].item()
        val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} / {max_iter} - Validation loss:"
                                f"{running_val_loss / (j + 1)}")
    running_val_loss /= len(val_loader)
    if prev_val_loss is None or prev_val_loss > running_val_loss:
        prev_val_loss = running_val_loss
        print('saving best model .... ')
        torch.save(copy.deepcopy(Net.state_dict()), "saved_path/magicpoint/magicpointBN.pt")
    writer.add_scalar('Loss', running_loss, n_iter + 1)
    writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
    writer.flush()
    n_iter += len(train_loader) * batch_size

