import torch
import yaml
from Synthetic_dataset_loader import SyntheticDataset
from model_loader import SuperPointNet, load_model, detector_post_processing, SuperPointNet_gauss2, SuperPointNetBatchNorm
from utils import flattenDetection, warpLabels, get_grid, filter_points, warp_image, nms_fast
from torchsummary import summary
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import argparse
warnings.simplefilter("ignore")


def repeatability(pts1, pts2, correct_distance=3):
    """
    Finds the no. of repeated points in both images given the keypoints
    param: pts1- keypoints listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: pts2 - keypoints listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: correct_distance - distance used to approximate whether a points is nearby or not
    """
    # variable names used here are identical with that of Superpoint paper
    N1 = pts1.shape[0]
    N2 = pts2.shape[0]
    count1, count2 = 1, 1
    pts1 = pts1[np.newaxis, ...]
    pts2 = pts2[:, np.newaxis, :]
    dist = np.linalg.norm(pts1 - pts2, axis=2)
    if N1 != 0:
        min_dist = np.min(dist, axis=1)
        count1 = np.sum(min_dist <= correct_distance)
    if N2 != 0:
        min_dist = np.min(dist, axis=0)
        count2 = np.sum(min_dist <= correct_distance)
    repeatability_metric = (count1 + count2) / (N1 + N2)
    return repeatability_metric


parser = argparse.ArgumentParser(description="This scripts helps to evaluate detector using different metrics")
parser.add_argument('--config', help='Path to config file',
                    default="../my_superpoint_pytorch/evaluation_config.yaml")
parser.add_argument('--epsilon',  help='threshold distance used to calculate repeatability', default=1, type=int)
args = parser.parse_args()
config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
size = config['data']['preprocessing']['resize']
epsilon = config['model']['epsilon']
model = SuperPointNetBatchNorm()
model_weights = torch.load(config['pretrained'])
model.load_state_dict(model_weights)
batch_size = config['model']['batch_size']
model.to('cuda')
summary(model, input_size=(1, size[0], size[1]))
data_set = SyntheticDataset(transform=None, task='test', **config)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
tqdm_bar = tqdm.tqdm(data_loader)
repeat_metric_list = []
model.train(mode=False)
for sample in tqdm_bar:
    tqdm_bar.set_description("Evaluation of Detector")
    if torch.cuda.is_available():
        sample['image'] = sample['image'].to('cuda')
    with torch.no_grad():
        output = model(sample['image'])
        semi = output['semi']
        keypoints = detector_post_processing(semi, conf_threshold=0.015, NMS_dist=1, limit_detection=600)
        gt_keypoint_2d = sample['label'].squeeze()
        gt_keypoint = np.nonzero(gt_keypoint_2d)
        repeat_metric_list.append(repeatability(gt_keypoint, keypoints))
        tqdm_bar.set_description(f"Repeatability -- {np.mean(repeat_metric_list)} ")

