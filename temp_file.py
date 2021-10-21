import cv2
from utils import inv_warp_image_batch, warp_points, warp_image_cv2, filter_points, my_inv_warp_image_batch
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import scipy
from torchgeometry.core import warp_perspective


def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(src.shape, dst.shape))

    def ax(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], dim=1)

    def ay(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    p.append(ax(src[:, 0], dst[:, 0]))
    p.append(ay(src[:, 0], dst[:, 0]))

    p.append(ax(src[:, 1], dst[:, 1]))
    p.append(ay(src[:, 1], dst[:, 1]))

    p.append(ax(src[:, 2], dst[:, 2]))
    p.append(ay(src[:, 2], dst[:, 2]))

    p.append(ax(src[:, 3], dst[:, 3]))
    p.append(ay(src[:, 3], dst[:, 3]))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)
    return M.view(-1, 3, 3)  # Bx3x3


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
    pts1_torch = torch.Tensor([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
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
    torch_hom = get_perspective_transform(torch.from_numpy(pts1).unsqueeze(0), torch.from_numpy(pts2).unsqueeze(0))
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]
    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    torch_hom = torch.from_numpy(homography)
    return homography, torch_hom


config_file = "joint_training.yaml"
with open(config_file) as file:
    config = yaml.full_load(file)
img = cv2.imread("C:/Users/Jeethesh/Desktop/Studienarbeit_articles/deep learning based co-registration/deep learning "
                 "based co-registration/Dataset/MSCOCO/Train/COCO_train2014_000000000009.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orig_image = np.copy(img)
# img = np.ones(shape=(24, 24), dtype=np.float32)*255
warped_pair_params = config['data']['augmentation']['homographic']['homographies']['params']
coords = np.stack(np.meshgrid(np.arange(img.shape[1]//8), torch.arange(img.shape[0]//8)), axis=2).astype(dtype=np.float32)
coords = np.transpose(coords, axes=(1, 0, 2))
coords = coords * 8 + 8 // 2
homography, torch_hom = sample_homography(np.array([img.shape[0], img.shape[1]]), shift=0, scaling=True, perspective=True,
                                          translation=True, patch_ratio=1, max_angle=0.785, rotation=True,
                                          perspective_amplitude_x=0.7, perspective_amplitude_y=0.7, allow_artifacts=True,
                                          scaling_amplitude=0.5)

# homography = np.eye(3).astype(np.float32)
# homography[0, 2] = 2
torch_inv = my_inv_warp_image_batch(torch.from_numpy(img[np.newaxis, np.newaxis, ...]).type(torch.float32),
                                    torch_hom.type(torch.float32)).unsqueeze(0)
torch_orig = my_inv_warp_image_batch(torch_inv.unsqueeze(0), torch.linalg.inv(torch_hom).type(torch.float32))
inv_img = warp_image_cv2(img, homography)
img = warp_image_cv2(inv_img, np.linalg.inv(homography))
warped_coord = warp_points(torch.from_numpy(coords.reshape([-1, 2])),
                           torch.linalg.inv(torch.from_numpy(homography)).type(torch.float32)).numpy()
coords = coords.reshape([-1, 1, 2]).astype(np.int)
orig_image[coords[:, 0, 1], coords[:, 0, 0]] = 255
warped_coord = warped_coord.reshape([1, -1, 2])
warp_mask = filter_points(warped_coord.squeeze(), (img.shape[1], img.shape[0]), indicesTrue=True)
warped_coords_masked = warped_coord.squeeze()[warp_mask[0]].astype(np.int)
inv_img[warped_coords_masked[:, 1], warped_coords_masked[:, 0]] = 255
torch_orig = torch_orig.numpy().squeeze()
torch_inv = torch_inv.numpy().squeeze()
torch_inv[warped_coords_masked[:, 1], warped_coords_masked[:, 0]] = 255
concat_img = np.hstack([orig_image, inv_img])
warped_coords_masked[:, 0] += img.shape[1]
coords_masked = coords.squeeze()[warp_mask[0]].astype(np.int)
concat_img = cv2.line(concat_img, coords_masked[0, :], warped_coords_masked[0, :], 255, 1)
concat_img = cv2.line(concat_img, coords_masked[-1, :], warped_coords_masked[-1, :], 255, 1)
concat_img = cv2.line(concat_img, coords_masked[-10, :], warped_coords_masked[-10, :], 255, 1)
concat_img = cv2.line(concat_img, coords_masked[-48, :], warped_coords_masked[-48, :], 255, 1)
concat_img = cv2.line(concat_img, coords_masked[20, :], warped_coords_masked[20, :], 255, 1)
concat_img = cv2.line(concat_img, coords_masked[100, :], warped_coords_masked[100, :], 255, 1)
# for i in range(coords_masked.shape[0]):
#     concat_img = cv2.line(concat_img, coords_masked[i, :], warped_coords_masked[i, :], 255, 1)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(orig_image, cmap='gray')
axes[1].imshow(inv_img, cmap='gray')
axes[2].imshow(img, cmap='gray')
plt.figure(figsize=(12, 12))
plt.imshow(concat_img, cmap='gray')
fig, axes = plt.subplots(1, 2)
axes[0].imshow(torch_orig, cmap='gray')
axes[1].imshow(torch_inv, cmap='gray')
plt.show()
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
