import cv2
import torch
from model_loader import SuperPointNetBatchNorm2, semi_to_heatmap
from utils import sample_homography
import numpy as np
import cv2
import matplotlib.pyplot as plt


def image_preprocess(file_name: str, size: tuple) -> np.ndarray:
    """
    :param file_name - path of the image
    :param size - dimension to resize to (width, height)
    """
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = (img / 255.0).astype(np.float32)
    return img


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = SuperPointNetBatchNorm2()
weight_dict = torch.load("../descriptorTrainingAfterIter2myloss.pt", map_location=torch.device(device))
Net.load_state_dict(weight_dict)
image_dir = "../Dataset/MSCOCO/Train/"
image1 = image_dir + "COCO_train2014_000000000025.jpg"
image1 = image_preprocess(image1, size=(640, 480))
homography = sample_homography(np.array([320, 216]), shift=0, scaling=True, perspective=True,
                               translation=True, patch_ratio=0.75, max_angle=0.785, rotation=True,
                               perspective_amplitude_x=0.1, perspective_amplitude_y=0.1, allow_artifacts=True,
                               scaling_amplitude=0.8).numpy().squeeze()
image2 = cv2.warpPerspective(image1, homography, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR, dsize=(640, 480))
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image1, cmap='gray')
axes[1].imshow(image2, cmap='gray')
plt.show()
Net.train(mode=False)
with torch.no_grad():
    out = Net(torch.from_numpy(image1[np.newaxis, np.newaxis, ...]) / 255.0)
    semi, desc = out['semi'], out['desc'].numpy()
    out_warp = Net(torch.from_numpy(image2[np.newaxis, np.newaxis, ...]) / 255.0)
    semi_warp, desc_warp = out_warp['semi'], out_warp['desc'].numpy()
    heatmap = semi_to_heatmap(semi).numpy()
    heatmap_warped = semi_to_heatmap(semi_warp).numpy()
    print('smothng')


