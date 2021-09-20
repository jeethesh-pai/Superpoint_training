import torch
import yaml
from Data_loader import TLSScanData, points_to_2D
from model_loader import SuperPointNet, load_model, detector_post_processing, SuperPointNet_gauss2
from utils import flattenDetection, warpLabels, get_grid, filter_points, warp_image, nms_fast
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
size = config['data']['preprocessing']['resize']  # width, height
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
# Net = SuperPointNet()
# Net.load_state_dict(weight_dict)
Net = SuperPointNet_gauss2()
weights = load_model(config['pretrained'], Net)
Net.load_state_dict(weights)
if torch.cuda.is_available():
    Net.cuda()
summary(Net, (1, size[1], size[0]), batch_size=1)  # shows the trained network architecture
if config['data']['generate_label']:
    label_path = config['data']['label_path']
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    tqdm_bar = tqdm.tqdm(data_loader)
    for sample in tqdm_bar:
        tqdm_bar.set_description(f"{args.split} Labels being created...")
        if torch.cuda.is_available():
            sample['image'] = sample['image'].to('cuda')
            sample['warped_image'] = sample['warped_image'].to('cuda').squeeze()  # squeeze the batch dimension as its 1
        # to store the homographic detections for averaging the response
        label = np.zeros((batch_size + numHomIter, size[1], size[0]), dtype=np.float32)
        with torch.no_grad():
            output = Net(sample['image'])
            semi, _ = output['semi'], output['desc']
            label[-1, :, :] = detector_post_processing(semi, ret_heatmap=True)
            warped_image = sample['warped_image'].unsqueeze(1)  # introduce the numHomIter as
            # batch size for prediction
            output_warped = Net(warped_image)
            semi_warped, _ = output_warped['semi'], output_warped['desc']
            for n in range(numHomIter):
                temp_heatmap = detector_post_processing(semi_warped[n, :, :, :], ret_heatmap=True)
                label[n, :, :] = warp_image(temp_heatmap, sample['homography'][0, n, :, :]).squeeze()
            label = np.sum(label, axis=0)  # average the heatmap over the no. of images.
            xs, ys = np.where(label >= config['model']['detection_threshold'])  # Confidence threshold.
            pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = label[xs, ys]
            H, W = label.shape
            pts, _ = nms_fast(pts, label.shape[0], label.shape[1], dist_thresh=1)
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]].astype(np.int16)  # Sort by confidence.
            # Remove points along border.
            bord = 4  # we consider 4 pixels from all the boundaries as rejected
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            if pts.shape[1] > config['model']['top_k']:
                pts = pts[:, :config['model']['top_k']]
            pts = list(zip(pts[1], pts[0]))  # saves points in the form pts =
            # (array(y axis coordinates), array(x axis coordinates))
            filename = os.path.join(label_path, split, sample['name'][0][:-4])
            plt.imshow(points_to_2D(np.asarray(pts, dtype=np.int16), H, W, img=sample['image'].to('cpu').numpy().squeeze() * 255), cmap='gray')
            plt.show()
            np.save(filename, pts)

