import torch
from model_loader import SuperPointNet
import numpy as np
from scipy.ndimage.interpolation import zoom
from cv2 import cv2

model = SuperPointNet()
pretrained = "../my_superpoint_pytorch/superpoint_v1.pth"
model.load_state_dict(torch.load(pretrained))
remote_model = torch.load("colab_log/detector_training_pt1.pt")
remote_weights = remote_model.state_dict()
colab_model = torch.load("colab_log/detector_training_pt2.pt")
pretrained_weights = model.state_dict()
colab_weights = colab_model.state_dict()
keys = pretrained_weights.keys()
for key in keys:
    p_weight = pretrained_weights[key]
    c_weight = colab_weights[key].to('cpu')
    r_weight = remote_weights[key].to('cpu')
    diff1 = torch.subtract(p_weight, c_weight).sum()
    diff2 = torch.subtract(c_weight, r_weight).sum()
    print(f"{key}:{diff1}, {diff2}")



