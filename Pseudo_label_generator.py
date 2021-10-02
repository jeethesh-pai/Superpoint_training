import torch
import yaml
from Data_loader import TLSScanData, points_to_2D
from model_loader import SuperPointNet, load_model, detector_post_processing, SuperPointNetBatchNorm, semi_to_heatmap
from utils import flattenDetection, warpLabels, get_grid, filter_points, warp_image, nms_fast, inv_warp_image_batch
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
parser.add_argument('--split', help='dataset split - train/validation', default='train')
args = parser.parse_args()

config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.load(path)
data_set = TLSScanData(transform=None, task=args.split, **config)
split = "Train" if args.split == 'train' else "Validation"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = config['model']['batch_size']
numHomIter = config['data']['augmentation']['homographic']['num']
det_threshold = config['model']['detection_threshold']  # detection threshold to threshold the detector heatmap
size = config['data']['preprocessing']['resize']  # width, height
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, prefetch_factor=2)
# include num_workers if trained on GPU
# Net = SuperPointNet()
# Net.load_state_dict(weight_dict)
Net = SuperPointNetBatchNorm()
weights = load_model(config['pretrained'], Net)
Net.load_state_dict(weights)
Net.to(device)
Net.train(mode=False)
summary(Net, (1, size[1], size[0]), batch_size=1)  # shows the trained network architecture
if config['data']['generate_label']:
    label_path = config['data']['label_path']
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    tqdm_bar = tqdm.tqdm(data_loader)
    for sample in tqdm_bar:
        tqdm_bar.set_description(f"{args.split} Labels being created...")
        agg_label = np.zeros_like(sample['image'])
        sample['image'] = sample['image'].to(device)
        sample['warped_image'] = sample['warped_image'].to(device).squeeze()  # squeeze the batch dimension as its 1
        sample['homography'] = sample['homography'].to(device)
        # to store the homographic detections for averaging the response
        # label = np.zeros((batch_size + numHomIter, size[1], size[0]), dtype=np.float32)
        with torch.no_grad():
            output = Net(sample['image'])
            semi, _ = output['semi'], output['desc']
            label = semi_to_heatmap(semi)
            # warped_image = sample['warped_image'].unsqueeze(1)
            # batch size for prediction
            for batch in range(batch_size):
                output_warped = Net(sample['warped_image'][batch, ...].unsqueeze(1))
                semi_warped, _ = output_warped['semi'], output_warped['desc']
                batch_heatmap = semi_to_heatmap(semi_warped)
                new_label = inv_warp_image_batch(batch_heatmap, sample['homography'][batch, ...].unsqueeze(0),
                                                 device=device).squeeze()
                new_label = torch.cat([new_label, label[batch, :, :].to(device).unsqueeze(0)], dim=0)
                new_label = torch.sum(new_label, dim=0)
                agg_label[batch, ...] = new_label.to('cpu').numpy()
            for batch in range(batch_size):
                label = agg_label[batch, ...].squeeze()
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
                filename = os.path.join(label_path, split, sample['name'][batch][:-4])
                plt.imshow(points_to_2D(np.asarray(pts, dtype=np.int16), H, W,
                                        img=sample['image'][batch, ...].to('cpu').numpy().squeeze() * 255), cmap='gray')
                plt.show()
                np.save(filename, pts)
