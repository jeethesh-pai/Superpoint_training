import torch
from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from photometric import ImgAugTransform
from utils import compute_valid_mask, sample_homography, warpLabels, warp_image
from numpy.linalg import inv


def points_to_2D(points: np.ndarray, H: int, W: int, img: np.ndarray) -> np.ndarray:
    labels = np.zeros((H, W))
    if len(points.shape) <= 1:
        return labels
    if img is not None:
        img = img / 255.0
        image_copy = np.copy(img)
        image_copy[points[:, 0], points[:, 1]] = 1
        return image_copy
    else:
        if points.shape[0] > 0:
            labels[points[:, 0], points[:, 1]] = 1
    return labels


class SyntheticDataset(Dataset):
    def __init__(self, transform=None, task='train', **config: dict):
        self.transform = transform
        if task == 'train':
            self.task = 'training'
        elif task == 'validation':
            self.task = 'validation'
        else:
            self.task = 'test'
        self.config = config
        self.image_dir_root = self.config['data']['root']
        self.shapes = os.listdir(self.image_dir_root)
        self.shapes_root = [os.path.join(self.image_dir_root, shape) for shape in self.shapes]
        self.image_list = [os.listdir(os.path.join(shape, 'images', self.task))
                           for shape in self.shapes_root]
        self.images = []
        for i, shape_root in enumerate(self.shapes_root):
            for image in self.image_list[i]:
                self.images.append(os.path.join(shape_root, 'images', self.task, image))
        self.photometric = self.config['data']['augmentation']['photometric']['enable']

    def __getitem__(self, index: int) -> dict:
        sample = {}
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # make sure label and image are of same size
        height, width = image.shape[0], image.shape[1]
        points = np.load(self.images[index].replace('images', 'points').replace('png', 'npy'))
        # points_y, points_x = point_erode(points)
        # points = np.asarray(list(zip(points_y, points_x)))
        points_2D = points_to_2D(np.asarray(points, dtype=np.int), height, width, img=image)
        points_2D = torch.tensor(points_2D, dtype=torch.float32)
        sample['label'] = points_2D
        if self.photometric:  # in photometric augmentations labels are unaffected
            aug = ImgAugTransform(**self.config['data']['augmentation'])
            image = aug(image)
        sample['image'] = torch.tensor(image / 255.0, dtype=torch.float32).unsqueeze(0)
        return sample

    def __len__(self):
        return len(self.images)




