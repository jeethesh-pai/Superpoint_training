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
warnings.simplefilter("ignore")

config_file_path = '../my_superpoint_pytorch/tls_scan_superpoint_config_file.yaml'
with open(config_file_path) as path:
    config = yaml.load(path)
train_set = TLSScanData(transform=None, task='train', **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['model']['batch_size'], shuffle=True)
val_set = TLSScanData(transform=None, task='validation', **config)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=True)
Net = SuperPointNet()
optimizer = optim.Adam(Net.parameters(), lr=config['model']['learning_rate'])
epochs = 0
Net, epoch, optimizer = load_model(config['pretrained'], Net, optimizer, epochs)
if torch.cuda.is_available():
    Net.cuda()
summary(Net, (1, 320, 240), batch_size=1)
if config['data']['generate_label']:
    label_path = config['data']['label_path']
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    for i, sample in tqdm.tqdm(enumerate(train_loader)):
        with torch.no_grad():
            output = Net(sample['image'].cuda().unsqueeze(0))
            keypoint, descriptor = output['semi'], output['desc']
            heatmap = flattenDetection(keypoint, True).cpu().numpy().squeeze()
            det_threshold = config['model']['detection_threshold']
            heatmap[heatmap < det_threshold] = 0  # as per confidence threshold given in Daniel De Tone paper.
            heatmap[heatmap >= det_threshold] = 1
            pts = np.nonzero(heatmap)
            pts = list(zip(pts[0], pts[1]))  # saves points in the form pts =
            # (array(y axis coordinates), array(x axis coordinates))
            filename = label_path + '/' + sample['name'][0][:-4]
            np.save(filename, pts)
else:  # training with given label and augmentations
    max_iter = config['train_iter']  # also called as epochs
    batch_size = config['model']['batch_size']
    homo_iter = config['data']['augmentation']['homographic']['num']
    n_iter = 0
    prev_val_loss = 0
    writer = SummaryWriter(log_dir="logs")
    writer.add_graph(Net, input_to_model=torch.ones(size=(2, 1, 320, 240)).cuda())
    while n_iter < max_iter:
        running_loss, batch_iou = 0, 0
        for count, sample in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0].imshow(sample['image'].numpy()[0, :, :, :].squeeze(), cmap='gray')
            # axes[0, 1].imshow(sample['warped_image'].numpy()[0, 1, :, :].squeeze(), cmap='gray')
            # axes[1, 0].imshow(sample['label'][0, :, :, :].numpy().squeeze(), cmap='gray')
            # axes[1, 1].imshow(sample['warped_label'][0, 1, :, :].numpy().squeeze(), cmap='gray')
            # plt.show()
            (_, loss), (_, iou) = Net.train_mode(sample).items()
            loss.backward()
            if count == 0:
                batch_iou = iou
            batch_iou = torch.add(batch_iou, iou) / 2
            running_loss += loss.item()
            optimizer.step()
        print(f'Training Loss for the epoch-{n_iter}: {running_loss / len(train_loader)}, IoU: {batch_iou}')
        val_loss, val_iou = 0, 0
        for count_val, sample_val in tqdm.tqdm(enumerate(val_loader)):
            with torch.no_grad():
                (_, curr_loss), (_, curr_iou) = Net.train_mode(sample_val).items()
                val_loss += curr_loss
                if count_val == 0:
                    val_iou = curr_iou
                val_iou = (val_iou + curr_iou) / 2
        if n_iter == 0:
            prev_val_loss = val_loss
            torch.save(Net, "saved_path/best_model.pt")
        if val_loss < prev_val_loss and n_iter > 0:
            print('saving best model...')
            torch.save(Net, "saved_path/best_model.pt")
        print(f'Val Loss for the epoch-{n_iter}: {val_loss / len(val_loader)}, Val_IoU: {val_iou}')
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', val_loss, n_iter + 1)
        writer.add_scalar('IoU', batch_iou, n_iter + 1)
        writer.add_scalar('Val_IoU', val_iou, n_iter + 1)
        writer.flush()
        n_iter += 1
    writer.close()
