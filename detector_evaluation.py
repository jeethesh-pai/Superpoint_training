import torch
import yaml
from Data_loader import TLSScanData, points_to_2D
from model_loader import SuperPointNet, detector_post_processing, SuperPointNetBatchNorm, ModelWrapper, semi_to_heatmap
from utils import flattenDetection, warpLabels, get_grid, filter_points, inv_warp_image_batch, getPtsFromHeatmap
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
    pts1 = pts1[np.newaxis, ...]
    pts2 = pts2[:, np.newaxis, :]
    dist = np.linalg.norm(pts1 - pts2, axis=2)
    count1, count2 = 0, 0
    loc_err1, loc_err2 = 0, 0
    if N1 != 0:
        min_dist = np.min(dist, axis=1)
        count1 = np.sum(min_dist <= correct_distance)
        loc_err1 = min_dist[min_dist <= correct_distance]
    if N2 != 0:
        min_dist = np.min(dist, axis=0)
        count2 = np.sum(min_dist <= correct_distance)
        loc_err2 = min_dist[min_dist <= correct_distance]
    if N1 + N2 > 0:
        repeatability_metric = (count1 + count2) / (N1 + N2)
        if count1 + count2 > 0:
            localization_err = 0
            if loc_err1 is not None:
                localization_err += (loc_err1.sum()) / (count1 + count2)
            if loc_err2 is not None:
                localization_err += (loc_err2.sum()) / (count1 + count2)
    else:
        repeatability_metric = 0
    return repeatability_metric, localization_err


parser = argparse.ArgumentParser(description="This scripts helps to evaluate detector using different metrics")
parser.add_argument('--config', help='Path to config file',
                    default="evaluation_config.yaml")
parser.add_argument('--epsilon',  help='threshold distance used to calculate repeatability', default=1, type=int)
args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
model = SuperPointNetBatchNorm()
size = config['data']['preprocessing']['resize']
epsilon = config['model']['epsilon']
nms = config['model']['nms']
model_weights = torch.load(config['pretrained'], map_location=device)
model.load_state_dict(model_weights)
thresh = config['model']['detection_threshold']
batch_size = config['model']['batch_size']
model.to(device)
summary(model, input_size=(1, size[0], size[1]))
data_set = TLSScanData(transform=None, task='test', **config)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
tqdm_bar = tqdm.tqdm(data_loader)
repeat_metric_list, loc_error_list = [], []
model.train(mode=False)
tqdm_bar.set_description("Evaluation of Detector")
for sample in tqdm_bar:
    W, H = config['data']['preprocessing']['resize']
    sample['image'] = sample['image'].to(device)
    sample['warped_image'] = sample['warped_image'].to(device).unsqueeze(0)  # introduce batch dimension
    with torch.no_grad():
        output = model(sample['image'])
        semi = output['semi']
        output_warped = model(sample['warped_image'])
        semi_warped = output_warped['semi']
        keypoints = detector_post_processing(semi, thresh, NMS_dist=nms, limit_detection=600)
        keypoints = np.stack([keypoints[0, :], keypoints[1, :]], axis=-1).astype(np.int)
        heatmap_warped = semi_to_heatmap(semi_warped)
        heatmap_warped = heatmap_warped.unsqueeze(0).unsqueeze(1)
        heatmap_unwarped = inv_warp_image_batch(heatmap_warped, sample['inv_homography'], device=device)
        keypoints_unwarped = getPtsFromHeatmap(heatmap_unwarped, thresh, nms_dist=nms,
                                               limit_detection=600)
        keypoints_unwarped = np.stack([keypoints_unwarped[0, :], keypoints_unwarped[1, :]], axis=-1).astype(np.int)
        # pts = list(zip(keypoints_unwarped[:, 1], keypoints_unwarped[:, 0]))
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(points_to_2D(np.asarray(pts, dtype=np.int16), H, W,
        #                             img=sample['image'][0, ...].to('cpu').numpy().squeeze() * 255), cmap='gray')
        # pts = list(zip(keypoints[:, 1], keypoints[:, 0]))
        # axes[1].imshow(points_to_2D(np.asarray(pts, dtype=np.int16), H, W,
        #                             img=sample['image'][0, ...].to('cpu').numpy().squeeze() * 255), cmap='gray')
        # plt.show()
        repeatable, loc_error = repeatability(keypoints, keypoints_unwarped, correct_distance=epsilon)
        repeat_metric_list.append(repeatable)
        loc_error_list.append(loc_error)
    tqdm_bar.set_description(f"Repeatability -- {np.mean(repeat_metric_list)}, "
                             f"Localization Error: {np.mean(loc_error_list)}")
print('Mean repeatability: ', np.mean(repeat_metric_list))





