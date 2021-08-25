from model_loader import SuperPointNet, load_model
from cv2 import cv2
import torch
import numpy as np
from utils import nn_match_descriptor
import matplotlib.pyplot as plt


def image_preprocess(file_name: str, size: tuple) -> np.ndarray:
    """
    :param file_name - path of the image
    :param size - dimension to resize to (width, height)
    """
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = (img / 255.0).astype(np.float32)
    return img


def offset_keypoint(keypoint: list, img1_shape: tuple) -> list:
    """
    This function offsets the keypoint of the second image for dense correspondence matching
    :param keypoint : list of cv2.KeyPoint objects of the second image whose x coordinates (width) needs to be offset
    :param img1_shape: shape of the first image, ideally the width of image is taken to offset the keypoint
    :return keypoint_new: offset keypoint which can be used for line drawing
    """
    point_convert = cv2.KeyPoint_convert(keypoint)
    point_convert[:, 0] = point_convert[:, 0] + img1_shape[0]
    keypoint_new = [cv2.KeyPoint(int(point_new[0]), int(point_new[1]), 1) for point_new in point_convert]
    return keypoint_new


def extract_superpoint_desc_keypoints(model: torch.nn.Module, img: str, size: tuple,
                                      conf_threshold=0.015, dist_thresh=4):
    img_gray = image_preprocess(img, size=size)
    keypoint, descriptor, heatmap = model.eval_mode(img_gray, conf_threshold, 640, 480, dist_thresh)
    keypoint = np.transpose(keypoint)
    keypoint = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in keypoint]
    return keypoint, descriptor


def draw_matches_superpoint(img1: str, img2: str, nn_thresh: float, size: tuple) -> \
        (np.ndarray, np.ndarray):
    """outputs an image which shows correspondence between two images with the help of superpoint descriptor and
    keypoint detector
    :param img1 - image 1 - 3 channel or 1 channel (image will be converted to grayscale)
    :param img2 - image 2
    :param size - dimension of the iamge to be resized to (height, width)
    :param nn_thresh - threshold use to reduce no. of outlier matched, ideally the nearest neighbour threshold
    :returns combined_image - image showing correspondence matches
    :returns kp_image - returns image with keypoints marked in it
    """
    keypoint1, descriptor1 = extract_superpoint_desc_keypoints(Net, img1, size=size)
    keypoint2, descriptor2 = extract_superpoint_desc_keypoints(Net, img2, size=size)
    img1, img2 = image_preprocess(img1, size), image_preprocess(img2, size)
    match = nn_match_descriptor(descriptor1, descriptor2, nn_thresh=nn_thresh)
    match_desc1_idx = np.array(match[0, :], dtype=int)  # descriptor 1 matches
    match_desc2_idx = np.array(match[1, :], dtype=int)  # descriptor 2 matches
    matched_keypoint1 = [keypoint1[idx] for idx in match_desc1_idx]
    matched_keypoint2 = [keypoint2[idx] for idx in match_desc2_idx]
    new_keypoint = offset_keypoint(keypoint2, size)
    combined_keypoint = np.concatenate([keypoint1, new_keypoint], axis=0)
    combined_image = cv2.hconcat([img1 * 255, img2 * 255])
    kp_image = np.copy(combined_image)
    kp_image = cv2.drawKeypoints(kp_image.astype(np.uint8), combined_keypoint, None, color=(0, 255, 0))
    match_point1 = cv2.KeyPoint_convert(matched_keypoint1)
    match_point2 = cv2.KeyPoint_convert(matched_keypoint2)
    H, inlier = cv2.findHomography(match_point1[:, [1, 0]], match_point2[:, [1, 0]], cv2.RANSAC)
    inlier = inlier.flatten()
    inlier_index = np.nonzero(inlier)
    match_point1 = np.squeeze(match_point1[inlier_index, :])
    match_point2 = np.squeeze(match_point2[inlier_index, :])
    match_point1 = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in match_point1]
    match_point2 = [cv2.KeyPoint(int(point[0]), int(point[1]), 1) for point in match_point2]
    match_point2 = offset_keypoint(match_point2, size)
    combined_keypoint = np.concatenate([match_point1, match_point2], axis=0)
    combined_image = cv2.drawKeypoints(combined_image.astype(np.uint8), combined_keypoint, None, color=(0, 255, 0))
    match_point2 = cv2.KeyPoint_convert(match_point2)
    match_point1 = cv2.KeyPoint_convert(match_point1)
    for i in range(len(match_point1)):
        point1_i = (int(match_point1[i][0]), int(match_point1[i][1]))
        point2_i = (int(match_point2[i][0]), int(match_point2[i][1]))
        combined_image = cv2.line(combined_image, point1_i, point2_i, color=(0, 255, 0),
                                  thickness=2)
    return combined_image, kp_image


image_dir = "../pytorch-superpoint/datasets/TLS_Train/Test/"
Net = SuperPointNet()
checkpoint_path = "Remote_log/best_model.pt"
Net = load_model(checkpoint_path, Net, optimizer=torch.optim.Adam(Net.parameters(), lr=1e-4), epoch=0)
image1 = image_dir + "IMG_2035.JPG"
image2 = image_dir + "rgb_syn_library_w1.jpg"
combined, key = draw_matches_superpoint(image1, image2, nn_thresh=0.7, size=(480, 640))
plt.imshow(combined)
plt.title('Correspondence Image')
plt.show()
plt.imshow(key)
plt.title('Total Keypoints found')
plt.show()
