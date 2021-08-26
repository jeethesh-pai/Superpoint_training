import numpy as np
import torch
from pathlib import Path
from cv2 import cv2
import torch.nn.functional as F
import torch.nn as nn
import csv
from numpy.random import normal
from numpy.random import uniform
from scipy.stats import truncnorm
import torchmetrics


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img

    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def save_path_formatter(args, parser):
    print("todo: save path")
    return Path('.')


def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = 0.5 + tensor.numpy() * 0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array


def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    # torch.save(net_state, save_path)
    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    torch.save(net_state, save_path / filename)
    print("save checkpoint to ", filename)
    pass


def load_checkpoint(load_path, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)
    # torch.save(net_state, save_path)
    checkpoint = torch.load(load_path / filename)
    print("load checkpoint from ", filename)
    return checkpoint
    pass


def saveLoss(filename, iter, loss, task='train', **options):
    # save_file = save_output / "export.txt"
    with open(filename, "a") as myfile:
        myfile.write(task + " iter: " + str(iter) + ", ")
        myfile.write("loss: " + str(loss) + ", ")
        myfile.write(str(options))
        myfile.write("\n")


def saveImg(img, filename):
    cv2.imwrite(filename, img)


def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


def loadConfig(filename):
    import yaml
    with open(filename, 'r') as f:
        config = yaml.load(f)
    return config


def append_csv(file='foo.csv', arr=[]):
    with open(file, 'a') as f:
        writer = csv.writer(f)
        if type(arr[0]) is list:
            for a in arr:
                writer.writerow(a)
                # writer.writerow(pre(a))
                # print(pre(a))
        else:
            writer.writerow(arr)


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
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y / 2).rvs(1)
        h_displacement_left = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        h_displacement_right = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1 * std_trunc, std_trunc, loc=1, scale=scaling_amplitude / 2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
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
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]
    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]
    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    return homography


def warpLabels(pnts, homography, H, W, filterTrue=True):
    """
    input:
        pnts: numpy
        homography: numpy - homography matrix
        filterTrue: filter out warped points
    output:
        warped_pnts: numpy
    """
    # https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315
    # see the above site for confusions in case of homography transformation
    # points we are receiving are of form (y, x) we need to change it to (x, y)
    # check the (x, y)
    # turn to homogenous for further calculations
    pnts = np.vstack((pnts[:, 1], pnts[:, 0], np.ones_like(pnts[:, 1])))
    warped_points = []
    # homography = torch.inverse(homography).squeeze()  # while coordinate transformation we use formula X' = H X
    if isinstance(homography, torch.Tensor):
        homography = homography.numpy().squeeze()
    if len(homography.shape) == 2:
        homography = homography[np.newaxis, :, :]
    for i in range(homography.shape[-3]):  # take dimension before H
        warped_pnts = (homography[i, :, :] @ pnts).transpose(1, 0)
        warped_pnts = warped_pnts[:, :2]
        if filterTrue:
            warped_pnts = filter_points(warped_pnts, np.asarray([W, H]))  # return as (y, x) coordinates
        warped_points.append(warped_pnts)
    return np.asarray(warped_points).squeeze()


def homography_scaling(homography, H, W):
    trans = np.array([[2. / W, 0., -1], [0., 2. / H, -1], [0., 0., 1.]])
    homography = np.linalg.inv(trans) @ homography @ trans
    return homography


def homography_scaling_torch(homography, H, W):
    trans = torch.tensor([[2. / W, 0., -1], [0., 2. / H, -1], [0., 0., 1.]])
    homography = (trans.inverse() @ homography @ trans)
    return homography


def filter_points(points, shape):
    #  check!
    x_warp, y_warp = points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)
    indices = np.where((x_warp >= 0) & (x_warp < shape[0]) & (y_warp >= 0) & (y_warp < shape[1]))
    points_warp = np.asarray(list(zip(y_warp[indices[0]], x_warp[indices[0]])))
    return points_warp


def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homographies: batched or not (shapes (B, 3, 3) and (...) respectively).
        device: gpu device or cpu
    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[1]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size * 3, 3)
    warped_points = homographies @ points.transpose(0, 1)
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords


def warp_image(img: np.ndarray, inv_homography: np.ndarray) -> np.ndarray:
    if not isinstance(inv_homography, np.ndarray):
        inv_homography = inv_homography.numpy()
    if len(inv_homography.shape) == 2:
        inv_homography = inv_homography[np.newaxis, :, :]
    canvas = np.zeros((inv_homography.shape[0], img.shape[0], img.shape[1]))
    coords = get_grid(img.shape[1], img.shape[0], True)
    x2, y2 = coords[0], coords[1]
    for i in range(inv_homography.shape[0]):
        warp_coords = inv_homography[i, :, :].squeeze() @ coords
        warp_coords = warp_coords.astype(np.int)
        x1, y1 = warp_coords[0, :], warp_coords[1, :]
        indices = np.where((x1 >= 0) & (x1 < img.shape[1]) & (y1 >= 0) & (y1 < img.shape[0]))
        x1pix, y1pix = x2[indices[0]], y2[indices[0]]
        x2pix, y2pix = x1[indices[0]], y1[indices[0]]
        canvas[i, y1pix.astype(np.int64), x1pix.astype(np.int64)] = img[y2pix, x2pix]
    return canvas


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=cell_size)
    label = pixel_unshuffle(labels)
    if not add_dustbin:
        return label
    else:
        if torch.cuda.is_available():
            label = torch.cat((label * 2, torch.ones(size=(batch_size, 1, Hc, Wc)).to('cuda')), dim=1)
        else:
            label = torch.cat((label * 2, torch.ones(size=(batch_size, 1, Hc, Wc))), dim=1)
        return label


def flattenDetection(semi):
    """
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    """
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]
    if batch:
        dense = nn.functional.softmax(semi, dim=1)  # [batch, 65, Hc, Wc]
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]
    else:
        dense = nn.functional.softmax(semi, dim=0)  # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)
    Hc, Wc = semi.shape[2:]
    H, W = Hc * 8, Wc * 8  # we know that all images fed to the Superpoint must be divisible by a factor of 8 for the
    # same purpose
    heatmap = torch.pixel_shuffle(nodust, upscale_factor=8)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    """
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    """

    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    # requires https://github.com/open-mmlab/mmdetection. 
    # Warning : BUILD FROM SOURCE using command MMCV_WITH_OPS=1 pip install -e
    # from mmcv.ops import nms as nms_mmdet 
    from torchvision.ops import nms
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = torch.nonzero(prob > min_prob).float()  # [N, 2]
    prob_nms = torch.zeros_like(prob)
    if pts.nelement() == 0:
        return prob_nms
    size = torch.tensor(size / 2.).cuda()
    boxes = torch.cat([pts - size, pts + size], dim=1)  # [N, 4]
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
        # indices, _ = nms(boxes, scores, iou, boxes.size()[0])
        # print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # proposals = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        # dets, indices = nms_mmdet(proposals, iou)
        # indices = indices.long()
        # indices = box_nms_retinaNet(boxes, scores, iou)
    pts = torch.index_select(pts, 0, indices)
    scores = torch.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros(1).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        image_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        inv_homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    if not isinstance(inv_homography, torch.Tensor):
        inv_homography = torch.tensor(inv_homography, dtype=torch.float32)
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]  # this is not exactly the batch size
    mask = np.ones((image_shape[0], image_shape[1]), dtype=np.uint8)
    mask = warp_image(mask, inv_homography)
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts / shape * 2 - 1
    return pts


def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts + 1) * shape / 2
    return pts


def detector_loss(target: torch.Tensor, output: torch.Tensor, mask=None) -> dict:
    """
    returns detector loss based on softmax activation as given in De Tone Paper.
    target: target label (Shape should be (Batch_size, 1, Hc * cell_size, Wc * cell_size)
    output: prediction from the network which is logit based and not activated by any functions
            shape of the output would be (Batch_size, 65, Hc, Wc)
    """
    if mask is None:
        mask = torch.ones((target.shape[2], target.shape[3]))
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    if len(target.shape) == 3:
        target = target.unsqueeze(1)
    labels3D = labels2Dto3D(target, 8, add_dustbin=True)
    with torch.no_grad():
        labels3D[labels3D == 2] = 1
        Softmax = torch.nn.Softmax(dim=1)
        softmax_output = Softmax(output)
        iou = torchmetrics.functional.iou(softmax_output, labels3D.to(torch.int), threshold=0.015)
    labels3D = torch.argmax(labels3D, dim=1)
    # dustbin is true because of softmax activation in output
    CE_loss = torch.nn.CrossEntropyLoss(reduction='none')
    entropy_loss = CE_loss(output, labels3D.to('cuda'))
    mask3D = labels2Dto3D(mask, cell_size=8, add_dustbin=False).float()
    mask3D = torch.prod(mask3D, dim=1).to('cuda')
    loss = (entropy_loss * mask3D).sum() / (mask3D.sum() + 1e-10) / target.shape[0]
    # add a small number to avoid division by zero
    return {'loss': loss, 'iou': iou}


def descriptor_loss_2(descriptor: torch.Tensor, descriptor_warped: torch.Tensor, homography: torch.Tensor,
                      lambda_d: float, margin_pos: float, margin_neg: float, threshold=8, valid_mask=None) -> \
        torch.Tensor:
    batch_size, Hc, Wc = descriptor_warped.shape[0], descriptor_warped.shape[2], descriptor_warped.shape[3]
    H, W = Hc * 8, Wc * 8
    valid_mask = torch.ones(size=(H, W)) if valid_mask is None else valid_mask
    if len(valid_mask.shape) == 3:
        valid_mask = valid_mask.unsqueeze(1)
    coords = get_grid(Hc, Wc, homogenous=False)  # coordinates of Hc, Wc  grid
    coords = coords * 8 + 8 // 2  # to get the center coordinates of respective expanded image with (H,W)
    warped_coord = warpLabels(coords.transpose(1, 0), homography, H, W, filterTrue=False)
    mask3D = labels2Dto3D(valid_mask, cell_size=8, add_dustbin=False).float()
    mask3D = torch.prod(mask3D, dim=1).to('cuda')
    loss = []
    for i in range(batch_size):  # batch_size here refer to the numHomoIter
        cell_dist = coords.reshape(-1, 2) - warped_coord[i, :, :]
        cell_dist = np.linalg.norm(cell_dist, axis=-1)
        mask = cell_dist <= threshold
        mask = mask.astype(np.float32)
        mask = torch.Tensor(mask.reshape(Hc, Wc)).to('cuda')
        desc_product = descriptor * descriptor_warped[i, :, :, :]
        desc_product_sum = desc_product.sum(dim=0)
        positive_corr = torch.max(margin_pos * torch.ones_like(desc_product_sum) - desc_product_sum,
                                  torch.zeros_like(desc_product_sum))
        negative_corr = torch.max(desc_product_sum - margin_neg * torch.ones_like(desc_product_sum),
                                  torch.zeros_like(desc_product_sum))
        loss_desc = (lambda_d * mask * positive_corr + (1 - mask) * negative_corr) * mask3D[i, :, :]
        loss.append(loss_desc.sum() / mask3D.sum())
    return sum(loss) / batch_size


def descriptor_loss(descriptors, descriptors_warped, homographies, mask_valid=None,
                    cell_size=8, lamda_d=250, device='cpu', descriptor_dist=4, **config):
    """
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    """

    # put to gpu
    homographies = homographies.to(device)
    # config
    lamda_d = lamda_d  # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    #####
    with torch.no_grad():
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image
        coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order
        warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        warped_coor_cells = torch.stack((warped_coor_cells[:, 1], warped_coor_cells[:, 0]), dim=1)  # (y, x) to (x, y)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]),
                                        dim=2)  # (batch, x, y) to (batch, y, x)

        shape_cell = torch.tensor([H // cell_size, W // cell_size]).type(torch.FloatTensor).to(device)
        warped_coor_cells = denormPts(warped_coor_cells, shape)
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])

        cell_distances = coor_cells - warped_coor_cells
        cell_distances = torch.norm(cell_distances, dim=-1)
        # check
        #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist  # 0.5 # trick

        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    # dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc, desc_len]

    # hinge loss
    positive_dist = torch.max(margin_pos - dot_product_desc, torch.tensor(0.).to(device))
    # positive_dist[positive_dist < 0] = 0
    negative_dist = torch.max(dot_product_desc - margin_neg, torch.tensor(0.).to(device))
    # negative_dist[neative_dist < 0] = 0
    # sum of the dimension

    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc * cell_size, Wc * cell_size)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
    # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    normalization = (batch_size * (mask_valid.sum() + 1) * Hc * Wc)
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum


def sumto2D(ndtensor):
    # input tensor: [batch_size, Hc, Wc, Hc, Wc]
    # output tensor: [batch_size, Hc, Wc]
    return ndtensor.sum(dim=1).sum(dim=1)


def precisionRecall_torch(pred, labels):
    offset = 10 ** -6
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = torch.sum(pred * labels) / (torch.sum(pred) + offset)
    recall = torch.sum(pred * labels) / (torch.sum(labels) + offset)
    if precision.item() > 1.:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <= 1. and precision.item() >= 0.
    return {'precision': precision, 'recall': recall}


def precisionRecall(pred, labels, thd=None):
    offset = 10 ** -6
    if thd is None:
        precision = np.sum(pred * labels) / (np.sum(pred) + offset)
        recall = np.sum(pred * labels) / (np.sum(labels) + offset)
    return {'precision': precision, 'recall': recall}


def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93
    # /fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!' % out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice


def nn_match_descriptor(desc1: torch.Tensor, desc2: torch.Tensor, nn_thresh: float):
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

