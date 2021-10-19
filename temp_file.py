import cv2
from utils import sample_homography, inv_warp_image_batch, warp_points
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

config_file = "joint_training.yaml"
with open(config_file) as file:
    config = yaml.full_load(file)
# img = cv2.imread("C:/Users/Jeethesh/Desktop/Studienarbeit_articles/deep learning based co-registration/deep learning "
#                  "based co-registration/Dataset/MSCOCO/Train/COCO_train2014_000000000009.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.ones(shape=(10, 10), dtype=np.float32)*255
warped_pair_params = config['data']['augmentation']['homographic']['homographies']['params']
coords = np.stack(np.meshgrid(np.arange(3), torch.arange(3)), axis=2).astype(dtype=np.float32)
coords = np.transpose(coords, axes=(1, 0, 2))
coords = coords * 8 + 8 // 2
# homography = sample_homography(np.array([2, 2]), shift=-1, **warped_pair_params)
homography = np.eye(3).astype(np.float32)
homography[0, 0] = np.cos(np.pi/4)
homography[1, 1] = np.cos(np.pi/4)
homography[0, 1] = -np.sin(np.pi/4)
homography[1, 0] = np.sin(np.pi/4)
inv_img = inv_warp_image_batch(torch.from_numpy(img[np.newaxis, np.newaxis, ...].astype(np.float32)),
                               mat_homo_inv=torch.linalg.inv(torch.from_numpy(homography))).numpy()
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img, cmap='gray')
axes[1].imshow(inv_img, cmap='gray')
plt.show()
warped_coord = warp_points(torch.from_numpy(coords.reshape([-1, 2])),
                           torch.from_numpy(homography).type(torch.float32)).numpy()
coords = coords.reshape([-1, 1, 2])
warped_coord = warped_coord.reshape([1, -1, 2])
norm = np.linalg.norm(coords - warped_coord, axis=-1)
norm_min = np.amin(norm, axis=-1)
true_correspondence = norm_min <= 8
true_points = np.arange(9)
norm_argmin = np.argmin(norm, axis=-1)
mask = np.zeros_like(norm)
mask[true_points[true_correspondence], norm_argmin[true_correspondence]] = 1.0
mask = norm <= 8
desc_1 = np.rand(size=(9, 5), dtype=torch.float32).unsqueeze(1)
desc_2 = np.rand(size=(9, 5), dtype=torch.float32).unsqueeze(2)
desc_prod = desc_1 * desc_2
desc_prod_2 = np.matmul(desc_1, desc_2)

print(warped_coord)
