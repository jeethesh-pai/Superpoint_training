import cv2
import os
import matplotlib.pyplot as plt
from utils import sample_homography
from photometric import ImgAugTransform
import numpy as np

image_dir = "../Dataset/MSCOCO/Validation/"
img = cv2.imread(image_dir + "COCO_train2014_000000000165.jpg", 0)
homography = sample_homography(np.array([img.shape[0], img.shape[1]]), shift=0, scaling=True, perspective=False,
                               translation=False, patch_ratio=0.85, max_angle=0.785, rotation=False,
                               perspective_amplitude_x=0.5, perspective_amplitude_y=0.5, allow_artifacts=True,
                               scaling_amplitude=0.6, translation_overflow=0.3)
# warped_image = cv2.warpPerspective(img, M=homography.numpy().squeeze(), dsize=(img.shape[1], img.shape[0]))
write_dir = "../Dataset/Augmentation_samples/"
photometric_config = {"photometric": {"enable": True,
                      "params": {"motion_blur": {"max_kernel_size": 15}}}}
                      # "random_contrast": {"strength_range": [0.5, 1.5]},
                      # "additive_gaussian_noise": {"stddev_range": [0, 10]},
                      # "additive_speckle_noise": {"prob_range": [0, 0.0035]},
                      # "additive_shade": {"transparency_range": [-0.5, 0.5]},
                      # "kernel_size_range": [100, 150], "motion_blur": {"max_kernel_size": 3}}
aug = ImgAugTransform(**photometric_config)
warped_image = aug(img)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img, cmap='gray')
axes[1].imshow(warped_image, cmap='gray')
plt.show()
cv2.imwrite(write_dir + "motion_blur.png", warped_image)

