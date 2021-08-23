from imgaug import augmenters as iaa
import numpy as np
from cv2 import cv2
from numpy.random import randint


class ImgAugTransform:
    def __init__(self, **config: dict):
        if config['photometric']['enable']:
            params = config['photometric']['params']
            aug_all = []
            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                aug_all.append(aug)
            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                aug = iaa.Sometimes(0.05, iaa.MotionBlur(change))
                aug_all.append(aug)
            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=change)
                aug_all.append(aug)

            self.aug = iaa.SomeOf(2, aug_all)
        else:
            self.aug = iaa.Sequential([iaa.Noop()])

    def __call__(self, img: np.ndarray):
        img = img.astype(np.uint8)
        img = self.aug.augment_image(img)
        return img


class customizedTransform:
    def __init__(self):
        pass

    def additive_shade(self, image, nb_ellipses=20, transparency_range=None,
                       kernel_size_range=None):
        if transparency_range is None:
            transparency_range = [-0.5, 0.8]
        if kernel_size_range is None:
            kernel_size_range = [250, 350]

        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        return shaded

    def __call__(self, img, **config):
        if config['photometric']['params']['additive_shade']:
            params = config['photometric']['params']
            img = self.additive_shade(img * 255, **params['additive_shade'])
        return img / 255