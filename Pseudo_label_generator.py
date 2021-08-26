import torch
import yaml
from Data_loader import TLSScanData, points_to_2D
from model_loader import SuperPointNet, load_model
from utils import flattenDetection, warpLabels
from torchsummary import summary
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import argparse
warnings.simplefilter("ignore")



parser = argparse.ArgumentParser(description="This scripts helps to generate pseudo ground truth label for Superpoint"
                                             "Homographic adaptation as mentioned in the paper")
parser.add_argument('--config', help='Path to config file',
                    default="../my_superpoint_pytorch/tls_scan_superpoint_config_file.yaml")
parser.add_argument('--split',  help='dataset split - train/validation', default='train')
args = parser.parse_args()


config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
data_set = TLSScanData(transform=None, task=args.split, **config)
split = "Train" if args.split == 'train' else "Validation"
batch_size = config['model']['batch_size']
numHomIter = config['data']['augmentation']['homographic']['num']
det_threshold = config['model']['detection_threshold']  # detection threshold to threshold the detector heatmap
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
Net = SuperPointNet()
Net = load_model(config['pretrained'], Net)
if torch.cuda.is_available():
    Net.cuda()
summary(Net, (1, 320, 240), batch_size=1)  # shows the trained network architecture
if config['data']['generate_label']:
    label_path = config['data']['label_path']
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    for sample in tqdm.tqdm(data_loader):
        if torch.cuda.is_available():
            sample['image'] = sample['image'].to('cuda')
            sample['warped_image'] = sample['warped_image'].to('cuda')
        with torch.no_grad():
            output = Net(sample['image'])
            semi, _ = output['semi'], output['desc']
            semi = flattenDetection(semi)
            semi[semi >= det_threshold] = 1
            semi[semi < det_threshold] = 0
            for batch in range(batch_size):
                warped_image = sample['warped_image'][batch, :, :, :].unsqueeze(1)  # introduce the numHomIter as
                # batch size for prediction
                output_warped = Net(warped_image)
                semi_warped, _ = output_warped['semi'], output_warped['desc']
                semi_warped = flattenDetection(semi_warped)
                semi_warped[semi_warped >= det_threshold] = 1  # threshold the heatmap based on the det_threshold
                semi_warped[semi_warped < det_threshold] = 0
                warped_label = np.zeros((numHomIter + 1, 432, 640), dtype=np.int8)  # store the homographic detections
                # for averaging the response
                for j in range(numHomIter):
                    pts = np.nonzero(semi_warped[j, :, :].to('cpu').numpy().squeeze())
                    pts = np.asarray(list(zip(pts[0], pts[1])))
                    warped_pts = warpLabels(pts, sample['inv_homography'][batch, j, :, :], 432, 640)
                    warped_label[j, :, :] = points_to_2D(warped_pts, 432, 640, img=None)
                    # fig, axes = plt.subplots(1, 2)
                    # axes[0].imshow(warped_label[j, :, :] * 255, cmap='gray')
                    # axes[1].imshow(semi_warped.to('cpu').numpy().squeeze()[j, :, :], cmap='gray')
                    # plt.show()
                # saving the non homographic detection also
                warped_label[j, :, :] = semi[batch, :, :, :].to('cpu').numpy().squeeze()
                label = np.mean(warped_label, axis=0)
                label[label >= 0.5] = 1
                label[label > 0.5] = 0
                pts = np.nonzero(label)
                pts = list(zip(pts[0], pts[1]))  # saves points in the form pts =
                # (array(y axis coordinates), array(x axis coordinates))
                filename = os.path.join(label_path, split, sample['name'][batch][:-4])
                np.save(filename, pts)
                # plt.imshow(label, cmap='gray')
                # plt.show()

