import cv2
from utils import inv_warp_image_batch, warp_points, warp_image_cv2, filter_points
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import scipy
from torchgeometry.core import warp_perspective


def sample_homography(shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi / 2,
                      allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.
    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        shift:
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """
    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])
    # Random perspective and affine perturbations

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        trunc_x = perspective_amplitude_x / 2
        trunc_y = perspective_amplitude_y / 2
        perspective_displacement = scipy.stats.truncnorm(-trunc_y, trunc_y, loc=0).rvs(1)
        h_displacement_left = scipy.stats.truncnorm(-trunc_x, trunc_x, loc=0).rvs(1)
        h_displacement_right = scipy.stats.truncnorm(-trunc_x, trunc_x, loc=0).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        trunc = scaling_amplitude / 2
        scales = scipy.stats.truncnorm(- trunc, trunc, loc=1).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled <= 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([np.random.uniform(-t_min[0], t_max[0], 1), np.random.uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        pts_centered = (pts2 - center).transpose(1, 0)
        rotated = np.matmul(rot_mat, pts_centered[np.newaxis, ...]) + center.T
        rotated = rotated.transpose(0, 2, 1)
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated <= 1.)
            valid = valid.prod(axis=(1, 2))
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]
    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]
    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    return homography


def Hpatches_preprocessing(image: np.ndarray, target_size: tuple, source_image_resolution: tuple):
    """
    :param image:- image to be processed
    :param target_size:- tuple containing target size (height, width)
    :param source_image_resolution - tuple (height, width) of original image.
    """
    fit_height = target_size[0]/image.shape[0] > target_size[1]/image.shape[1]
    scale = target_size[0] / image.shape[0] if fit_height else target_size[1] / image.shape[1]

    if source_image_resolution is not None:
        scale = target_size[0] / source_image_resolution[0] if fit_height else target_size[1] / source_image_resolution[1]

    new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    image_resized = cv2.resize(image, new_size)

    if fit_height:
        image_cropped = image_resized[:, :target_size[1]]
    else:
        image_cropped = image_resized[:target_size[0], :]
    return image_resized, image_cropped


def adapt_homography_to_preprocessing(source_size: tuple, hom: np.ndarray, target_size: tuple):
    """
    returns corrected homography for the new size
    :param source_size - size of the original image tuple - (height, width)
    :param target_size -  size of the target image tuple - (height, width)
    :param hom - homography matrix which is to be corrected
    :param warped_source_size - size of the original warped image (height,width)
    """
    scale = np.amax(np.divide(target_size, source_size))
    # scale_warped = np.amax(np.divide(target_size, warped_source_size))
    fit_height = (target_size[0] / source_size[0]) > (target_size[1] / source_size[1])
    padding_x, padding_y = 0, 0
    if fit_height:
        padding_x = int((source_size[1] * scale - target_size[1]) / 2)
    else:
        padding_y = int((source_size[0] * scale - target_size[0]) / 2)
    t_mat = np.vstack([[1, 0, padding_x], [0, 1, padding_y], [0, 0, 1]])
    # warped_padding_x = int((warped_source_size[1] * scale_warped - target_size[1]) / 2)
    # warped_padding_y = int((warped_source_size[0] * scale_warped - target_size[0]) / 2)
    # warped_t_mat = np.vstack([[1, 0, -warped_padding_x], [0, 1, -warped_padding_y], [0, 0, 1]])
    downscale = np.diag([1 / scale, 1 / scale, 1.0]).astype(np.float32)
    upscale = np.diag([scale, scale, 1.0]).astype(np.float32)
    # scale_matrix = np.array([[1, 1, scale], [1, 1, scale], [1 / scale, 1 / scale, 1]])
    actual_hom = upscale @ hom @ downscale @ t_mat
    return actual_hom


img_dir = "../Dataset/HPatches/v_talent/"
img = cv2.imread(img_dir + "1.ppm")
img_2 = cv2.imread(img_dir + "2.ppm")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
# homography = sample_homography(np.array([img.shape[0], img.shape[1]]), shift=0, scaling=True, perspective=True,
#                                translation=True, patch_ratio=0.85, max_angle=0.785, rotation=True,
#                                perspective_amplitude_x=0.5, perspective_amplitude_y=0.5, allow_artifacts=True,
#                                scaling_amplitude=0.5)
homography = np.loadtxt(img_dir + "H_1_2")
img_resized, img_cropped = Hpatches_preprocessing(img, (480, 640), None)
img2_resized, img2_cropped = Hpatches_preprocessing(img_2, (480, 640), img.shape)
new_hom = adapt_homography_to_preprocessing(img_2.shape, homography, img2_resized.shape)
warped_img = cv2.warpPerspective(img_resized, new_hom, dsize=(img2_resized.shape[1], img2_resized.shape[0]),
                                 flags=cv2.INTER_LINEAR)
new_overlay = cv2.addWeighted(warped_img, 0.5, img2_resized, 0.5, 0)
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(warped_img, cmap='gray')
axes[0, 1].imshow(img_resized, cmap='gray')
axes[1, 0].imshow(img2_resized, cmap='gray')
axes[1, 1].imshow(new_overlay, cmap='gray')
plt.show()
