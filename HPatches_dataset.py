import torch
from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from photometric import ImgAugTransform
from utils import compute_valid_mask, sample_homography, warpLabels, warp_image, inv_warp_image_batch
from numpy.linalg import inv
import re


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


class HPatches(Dataset):
    def __init__(self, transform=None, **config: dict):
        super(HPatches, self).__init__()
        self.transform = transform
        self.config = config
        self.image_folder = self.config['data']['root']
        self.image_subfolders = os.listdir(self.image_folder)
        self.image_path = [os.path.join(self.image_folder, folder) for folder in self.image_subfolders]
        self.resize_shape = self.config['data']['preprocessing']['resize']
        self.photometric = self.config['data']['augmentation']['photometric']['enable']
        self.homographic = self.config['data']['augmentation']['homographic']['enable']

    def __getitem__(self, index: int) -> dict:
        sample = {}
        image_list = [os.path.join(self.image_path[index], file_name) for file_name in
                      os.listdir(self.image_path[index]) if file_name[-3:] == 'ppm']
        images = [cv2.imread(image) for image in image_list]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        images = np.asarray([cv2.resize(image, tuple(self.resize_shape), cv2.INTER_AREA)
                             for image in images])[:, np.newaxis, ...]
        height, width = images.shape[-2], images.shape[-1]
        # if self.photometric:  # in photometric augmentations labels are unaffected
        #     aug = ImgAugTransform(**self.config['data']['augmentation'])
        #     image = torch.from_numpy(aug(image))
        num_iter = len(image_list) - 1  # homography list
        # use inverse of homography as we have initial points which needs to be homographically augmented
        homographies = np.zeros(shape=(num_iter, 3, 3), dtype=np.float32)
        homographies[0, ...] = np.eye(N=3, dtype=np.float32)
        for i in range(1, num_iter):
            homographies[i, ...] = np.loadtxt(os.path.join(self.image_path[index], f'H_1_{i + 1}'))
        inv_homography = torch.as_tensor([inv(homographies[i, ...]) for i in range(num_iter)], dtype=torch.float32)
        sample['warped_mask'] = compute_valid_mask(torch.tensor([height, width]), inv_homography=inv_homography)
        sample['homography'] = torch.from_numpy(homographies).type(torch.float32)
        sample['image'] = (torch.from_numpy(images) / 255.0).type(torch.float32)
        sample['inv_homography'] = inv_homography
        sample['name'] = ["_".join(re.split(r'[^\w]', file)[-3:-1]) for file in image_list]
        return sample

    def __len__(self):
        return len(self.image_path)
