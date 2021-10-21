import torch
from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from photometric import ImgAugTransform
from utils import sample_homography, inv_warp_image_batch, warp_points, filter_points_batch


def points_to_2D(points: np.ndarray, H: int, W: int, img=None) -> np.ndarray:
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


class TLSScanData(Dataset):
    def __init__(self, transform=None, task='train', **config: dict):
        super(TLSScanData, self).__init__()
        self.transform = transform
        self.config = config
        if task == 'train':
            self.task = 'Train'
        elif task == 'test':
            self.task = 'Test'
        else:
            self.task = 'Validation'
        if self.config['data'].get('labels', False):
            self.label_path = os.path.join(self.config['data']['label_path'], self.task)
        self.image_path = os.path.join(self.config['data']['root'], self.task)
        self.image_list = os.listdir(self.image_path)
        self.resize_shape = self.config['data']['preprocessing']['resize']
        self.photometric = self.config['data']['augmentation']['photometric']['enable']
        self.homographic = self.config['data']['augmentation']['homographic']['enable']
        self.warped_pair_params = self.config['data']['augmentation']['homographic']['homographies']['params']
        self.sample_homography = sample_homography

    def __getitem__(self, index: int) -> dict:
        sample = {}
        image = cv2.imread(os.path.join(self.image_path, self.image_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # make sure label and image are of same size
        points, points_2D = None, None  # initialization to avoid warning
        image = cv2.resize(image, tuple(self.resize_shape), cv2.INTER_AREA)
        height, width = image.shape[0], image.shape[1]
        if self.config['data'].get('labels', False):
            points = np.load(os.path.join(self.label_path, self.image_list[index][:-3] + 'npy'))
            points_2D = points_to_2D(points, height, width, img=None)
            points_2D = torch.tensor(points_2D, dtype=torch.float32).unsqueeze(0)
            sample['label'] = points_2D
        if self.photometric:  # in photometric augmentations labels are unaffected
            aug = ImgAugTransform(**self.config['data']['augmentation'])
            image = aug(image)
        sample['valid_mask'] = torch.ones_like(torch.from_numpy(image), dtype=torch.float32).unsqueeze(0)
        image = torch.from_numpy(image).type(torch.float32)
        if self.homographic:
            num_iter = self.config['data']['augmentation']['homographic']['num']
            # use inverse of homography as we have initial points which needs to be homographically augmented
            homographies = torch.stack([self.sample_homography(np.array([image.shape[0], image.shape[1]]),
                                                               shift=0, **self.warped_pair_params)
                                        for i in range(num_iter)]).type(torch.float32)  # actual homography
            if torch.prod(torch.linalg.det(homographies)) == 0:
                while torch.prod(torch.linalg.det(homographies)) != 0:
                    homographies = torch.stack([self.sample_homography(np.array([image.shape[0], image.shape[1]]),
                                                                       shift=0, **self.warped_pair_params)
                                                for i in range(num_iter)]).type(torch.float32)
            inv_homography = torch.linalg.inv(homographies)
            warped_image = torch.cat([image.unsqueeze(0)]*num_iter, dim=0) / 255.0
            sample['homography'] = homographies
            sample['inv_homography'] = inv_homography
            sample['warped_image'] = inv_warp_image_batch(warped_image.unsqueeze(1), mode='bilinear',
                                                          mat_homo_inv=sample['homography']).unsqueeze(0)
            sample['warped_mask'] = inv_warp_image_batch(torch.cat([sample['valid_mask'].unsqueeze(0)]*num_iter, dim=0),
                                                         sample['homography']).unsqueeze(0)
            if self.config['data'].get('labels', False):
                # warped_points_2D = sample['warped_image'].clone()
                warped_points_2D = torch.zeros_like(warped_image, dtype=torch.float32)
                points = torch.from_numpy(points).type(torch.float32)
                points = torch.vstack([points[:, 1], points[:, 0]]).transpose(1, 0)
                warped_points = warp_points(points, sample['inv_homography']).type(torch.int64)
                trueIdx = filter_points_batch(warped_points, (width, height))
                warped_points_2D[trueIdx[0], warped_points[trueIdx[0], trueIdx[1], 1],
                                 warped_points[trueIdx[0], trueIdx[1], 0]] = 1.0
                sample['warped_label'] = warped_points_2D.unsqueeze(0)
        sample['image'] = (image.unsqueeze(0) / 255.0)
        sample['name'] = self.image_list[index]
        return sample

    def __len__(self):
        if self.config['data'].get('labels', False):
            assert len(os.listdir(self.label_path)) == len(os.listdir(self.image_path)), \
                "Labels and image files mismatch!!"
        return len(os.listdir(self.image_path))
