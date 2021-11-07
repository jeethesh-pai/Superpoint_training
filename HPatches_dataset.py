import torch
from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from photometric import ImgAugTransform
from utils import compute_valid_mask, sample_homography, warpLabels, inv_warp_image_batch
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
        self.resize_shape = self.config['data']['preprocessing']['resize']  # width, height
        self.photometric = self.config['data']['augmentation']['photometric']['enable']
        self.homographic = self.config['data']['augmentation']['homographic']['enable']

    @staticmethod
    def preprocess(image: np.ndarray, target_size: tuple, source_image_resolution: tuple):
        """
        :param image:- image to be processed
        :param target_size:- tuple containing target size (height, width)
        :param source_image_resolution - tuple (height, width) of original image.
        """
        fit_height = target_size[0] / image.shape[0] > target_size[1] / image.shape[1]
        scale = target_size[0] / image.shape[0] if fit_height else target_size[1] / image.shape[1]
        if source_image_resolution is not None:
            scale = target_size[0] / source_image_resolution[0] if fit_height else target_size[1] / \
                                                                                   source_image_resolution[1]
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image_resized = cv2.resize(image, new_size)
        if fit_height:
            image_cropped = image_resized[:, :target_size[1]]
        else:
            image_cropped = image_resized[:target_size[0], :]
        return image_resized, image_cropped

    def __getitem__(self, index: int) -> dict:
        sample = {}
        image_list = [os.path.join(self.image_path[index], file_name) for file_name in
                      os.listdir(self.image_path[index]) if file_name[-3:] == 'ppm']
        images = [cv2.imread(image, 0) for image in image_list]
        size_original = [images[i].shape for i in range(len(images))]
        sample['change'] = "viewpoint" if re.split(r'[^\w]', image_list[0])[-3][0] == 'v' else "illumination"
        if sample['change'] == "illumination":
            images = np.asarray([cv2.resize(image, tuple(self.resize_shape), cv2.INTER_AREA)
                                 for image in images])[:, np.newaxis, ...]
        height_target, width_target = self.resize_shape[1], self.resize_shape[0]
        num_iter = len(image_list)  # homography list
        # use inverse of homography as we have initial points which needs to be homographically augmented
        homographies = np.concatenate([np.eye(3, dtype=np.float32)[np.newaxis, ...]]*num_iter, axis=0)
        if sample['change'] == "viewpoint":
            base_image = images[0]
            cropped_images = []
            scale_original = max(height_target / size_original[0][0], width_target / size_original[0][1])
            new_size = (int(size_original[0][1] * scale_original), int(size_original[0][0] * scale_original))
            # base_image_resized, base_image_cropped = self.preprocess(base_image, None)
            base_image_resized, base_image_cropped = self.preprocess(base_image,
                                                                                 (self.resize_shape[1], self.resize_shape[0]), None)
            cropped_images.append(base_image_cropped / 255.0)
            images[0] = torch.from_numpy(base_image_resized / 255).type(torch.float32)
            for i in range(1, num_iter):
                warped_image = images[i]
                homography = np.loadtxt(os.path.join(self.image_path[index], f'H_1_{i + 1}'))
                # warped_resized, warped_cropped = self.preprocess(warped_image, scale_original)
                warped_resized, warped_cropped = self.preprocess(warped_image,
                                                                 (self.resize_shape[1], self.resize_shape[0]),
                                                                 base_image.shape)
                cropped_images.append(warped_cropped)
                images[i] = torch.from_numpy(warped_resized / 255.0).type(torch.float32)
                # homographies[i, ...] = self.adapt_homography_to_resize(homography, size_original[i],
                #                                                        warped_resized.shape)
                homographies[i, ...] = self.adapt_homography_to_preprocessing(warped_image.shape, homography,
                                                                              warped_resized.shape)
                sample['cropped_image'] = cropped_images
        inv_homography = torch.as_tensor([inv(homographies[i, ...]) for i in range(num_iter)], dtype=torch.float32)
        sample['homography'] = torch.from_numpy(homographies).type(torch.float32)
        sample['image'] = images
        sample['inv_homography'] = inv_homography
        sample['name'] = ["_".join(re.split(r'[^\w]', file)[-3:-1]) for file in image_list]
        return sample

    @staticmethod
    def adapt_homography_to_preprocessing(source_size: tuple, hom: np.ndarray, target_size: tuple):
        """
        returns corrected homography for the new size
        :param source_size - size of the original image tuple - (height, width)
        :param target_size -  size of the target image tuple - (height, width)
        :param hom - homography matrix which is to be corrected
        """
        scale = np.amax(np.divide(target_size, source_size))
        fit_height = (target_size[0] / source_size[0]) > (target_size[1] / source_size[1])
        padding_x, padding_y = 0, 0
        if fit_height:
            padding_x = int((source_size[1] * scale - target_size[1]) / 2)
        else:
            padding_y = int((source_size[0] * scale - target_size[0]) / 2)
        t_mat = np.vstack([[1, 0, padding_x], [0, 1, padding_y], [0, 0, 1]])
        downscale = np.diag([1 / scale, 1 / scale, 1.0]).astype(np.float32)
        upscale = np.diag([scale, scale, 1.0]).astype(np.float32)
        actual_hom = upscale @ hom @ downscale @ t_mat
        return actual_hom

    def __len__(self):
        return len(self.image_path)
