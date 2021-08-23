from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from utils import sample_homography
import yaml


def points_to_2D(points: np.ndarray, H: int, W: int, img=None) -> np.ndarray:
    labels = np.zeros((H, W))
    if img is not None:
        img = img / 255.0
        image_copy = np.copy(img)
        image_copy[points[:, 0], points[:, 1]] = 1
        return image_copy
    else:
        h = points.shape[0]
        if points.shape[0] > 0:
            labels[points[:, 0], points[:, 1]] = 1
    return labels


def point_erode(points: np.ndarray) -> np.ndarray:
    y_coord, x_coord = points[:, 0], points[:, 1]
    pos = []
    for i in range(len(y_coord) - 1):
        diff_y = y_coord[i + 1:] - y_coord[i]
        diff_x = x_coord[i + 1:] - x_coord[i]
        dist = np.linalg.norm(np.vstack((diff_x, diff_y)), None, axis=0)
        pos_to_delete = np.where(np.logical_and(dist > 0, dist < 5))
        if len(pos_to_delete) > 0:
            pos.extend(pos_to_delete[0] + i + 1)
    y_coord = np.delete(y_coord, np.unique(pos), None)
    x_coord = np.delete(x_coord, np.unique(pos), None)
    return y_coord, x_coord


def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords


config_file = '../my_superpoint_pytorch/tls_scan_superpoint_config_file.yaml'
with open(config_file) as file:
    config = yaml.load(file)

homography = sample_homography(np.array([2, 2]), shift=-1,
                               **config['data']['augmentation']['homographic']['homographies']['params'])

image_path = '../pytorch-superpoint/datasets/TLS_Train/Train/IMG_9147.JPG'
label_path = '../pytorch-superpoint/datasets/TLS_Train/Label/Train/'
image = cv2.imread(image_path)
image = cv2.resize(image, (240, 320))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
label = label_path + image_path[-12:-3] + 'npy'
pnts = np.load(label)
y, x = point_erode(pnts)
pnts_homogenous = np.vstack((x, y, np.ones((1, y.shape[0]))))
points2d = points_to_2D(np.asarray(list(zip(y, x))), image.shape[0], image.shape[1], image)
homography_mat_translate = np.array([[1, 0, 20], [0, 1, 20], [0, 0, 1]])
homography_mat_scale = np.array([[2, 0, 0], [0, 2, 0], [1.2, 1.5, 1]])  # projective added
homography_mat_rotate = np.array([[np.cos(np.radians(45)), np.sin(np.radians(45)), 0],
                                  [-np.sin(np.radians(45)), np.cos(np.radians(45)), 0],
                                  [0, 0, 1]])
new_homograpy = homography_mat_rotate @ homography_mat_scale @ homography_mat_translate
coords = get_grid(image.shape[1], image.shape[0], True)
matrix = np.array([[1.2, 0, 100],
                   [0, 1.2, 50],
                   [0.003, 0.003, 1]])
x2, y2 = coords[0], coords[1]
warp_coords = np.linalg.inv(homography) @ coords
warp_coords = warp_coords.astype(np.int)
x1, y1 = warp_coords[0, :], warp_coords[1, :]
indices = np.where((x1 >= 0) & (x1 < image.shape[1]) & (y1 >= 0) & (y1 < image.shape[0]))
x1pix, y1pix = x2[indices[0]], y2[indices[0]]
x2pix, y2pix = x1[indices[0]], y1[indices[0]]
canvas = np.zeros_like(image)
canvas[y1pix.astype(np.int64), x1pix.astype(np.int64)] = image[y2pix, x2pix]
warp_points = (homography @ pnts_homogenous).transpose(1, 0)
warp_points = warp_points[:, :2]
x_warp, y_warp = warp_points[:, 0].astype(np.int64), warp_points[:, 1].astype(np.int64)
indices = np.where((x_warp >= 0) & (x_warp < image.shape[1]) & (y_warp >= 0) & (y_warp < image.shape[0]))
pnts_warp = np.asarray(list(zip(y_warp[indices], x_warp[indices])))
points2d_warp = points_to_2D(pnts_warp, image.shape[0], image.shape[1], canvas)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(points2d, cmap='gray')
axes[1].imshow(points2d_warp, cmap='gray')
plt.show()
