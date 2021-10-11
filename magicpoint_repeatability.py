import torch
import yaml
from Synthetic_dataset_loader import SyntheticDataset
from model_loader import SuperPointNet, load_model, detector_post_processing, SuperPointNet_gauss2, SuperPointNetBatchNorm
from utils import flattenDetection, warpLabels, get_grid, filter_points, warp_image, nms_fast
from torchsummary import summary
from Data_loader import TLSScanData
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


def localization_error(prediction, gt, prob_threshold=0.5, correct_distance=3):
    """
    finds out the localization error of the detected keypoints
    param: pts1: ground truth keypoints in the form [(y1, x1), (y2, x2), y3, x3), ...]
    param: pts2: predicted keypoint after post processing same as above form
    param: correct_distance: distance which qualifies a point to be regarded as correctly classified
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    gt = np.where(gt)
    pts_gt = np.stack([gt[0], gt[1]], axis=-1)
    pts_gt = pts_gt[np.newaxis, ...]
    mask = np.where(prediction > prob_threshold)
    pts_pred = np.array(mask).T
    pts_pred = pts_pred[:, np.newaxis, :]
    dist = np.linalg.norm(pts_pred - pts_gt, axis=2)
    if dist.shape[0] == 0 and dist.shape[1] == 0:
        return 0.0
    if dist.shape[0] == 0 or dist.shape[1] == 0:
        return correct_distance + 1
    dist = np.min(dist, axis=1)
    correctness = dist[np.less_equal(dist, correct_distance)]
    return np.mean(correctness)


def compute_tp_fp(prediction, gt, prob_threshold=[0.015, 0.5], correct_distance=3):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    gt = np.where(gt)
    pts_gt = np.stack([gt[0], gt[1]], axis=-1)
    pts_gt = pts_gt[np.newaxis, ...]
    tp_thresh, fp_thresh = [], []
    for threshold in prob_threshold:
        mask = np.where(prediction > threshold)  # remove near zero entries
        prob = prediction[mask]
        pts_pred = np.array(mask).T
        sort_idx = np.argsort(-prob)
        pts_pred = pts_pred[sort_idx]
        pts_pred = pts_pred[:, np.newaxis, :]
        dist = np.linalg.norm(pts_pred - pts_gt, axis=2)
        matches = np.less_equal(dist, correct_distance)
        tp = []
        matched = np.zeros(pts_gt.shape[1])
        for m in matches:
            correct = np.any(m)
            if correct:
                gt_idx = np.argmax(m)
                tp.append(not matched[gt_idx])
                matched[gt_idx] = 1
            else:
                tp.append(False)
        tp = np.array(tp, bool)
        fp = np.logical_not(tp)
        tp = np.sum(tp)
        fp = np.sum(fp)
        tp_thresh.append(tp)
        fp_thresh.append(fp)
    n_gt = pts_gt.shape[1]
    return tp_thresh, fp_thresh, n_gt


parser = argparse.ArgumentParser(description="This scripts helps to evaluate detector using different metrics")
parser.add_argument('--config', help='Path to config file',
                    default="evaluation_config.yaml")
parser.add_argument('--epsilon',  help='threshold distance used to calculate repeatability', default=1, type=int)
args = parser.parse_args()
config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
size = config['data']['preprocessing']['resize']
epsilon = config['model']['epsilon']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SuperPointNetBatchNorm()
model_weights = torch.load(config['pretrained'], map_location=device)
model.load_state_dict(model_weights)
batch_size = config['model']['batch_size']
model.to(device)
summary(model, input_size=(1, size[0], size[1]))
# data_set = SyntheticDataset(transform=None, task='test', **config)
data_set = TLSScanData(transform=None, task='val', **config)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
tqdm_bar = tqdm.tqdm(data_loader)
thresh = [0.015, 0.15, 0.25, 0.35, 0.45, 0.5, 0.65, 0.75]
repeat_metric_list, loc_error = [], []
fp_array, tp_array = np.zeros_like(thresh), np.zeros_like(thresh)
gt = 0
count = 0
model.train(mode=False)
for count, sample in enumerate(tqdm_bar):
    sample['image'] = sample['image'].to(device)
    if count > 1000:
        break
    with torch.no_grad():
        output = model(sample['image'])
        semi = output['semi']
        pred = detector_post_processing(semi, ret_heatmap=True)
        # plt.imshow(pred, cmap='gray')
        # plt.show()
        # repeat_metric_list.append(repeatability(gt_keypoint, keypoints))
        # loc_error.append(localization_error(pred, sample['label'].numpy().squeeze()))
        true_positive, false_positive, ground_truth = compute_tp_fp(pred, sample['label'].numpy().squeeze(),
                                                                    prob_threshold=thresh, correct_distance=5)
        tp_array += true_positive
        fp_array += false_positive
        gt += ground_truth
    tqdm_bar.set_description(f"Evaluation of Detector - mAP check")
    count += 1
recall = tp_array / gt
precision = tp_array / (tp_array + fp_array)
f1 = 2 * np.divide(precision * recall, (precision + recall))
sortIdx = np.argsort(recall)
recall = recall[sortIdx]
precision = precision[sortIdx]
plt.plot(recall, precision)
plt.show()
recall = np.concatenate([[0], recall])
AP = np.sum(precision * (recall[1:] - recall[:-1]))
print(f"Threshold range:{thresh}")
print(f'Max. F1 score for the threshold: {thresh[np.argmax(f1)]} from F1 score {f1}')
print(AP)


