import torch
import os
import numpy as np
import cv2
from model_loader import SuperPointNetBatchNorm, detector_post_processing
from Data_loader import points_to_2D
import matplotlib.pyplot as plt


def image_processing(file_name: str) -> torch.Tensor:
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = torch.from_numpy(image) / 255.0
    return image


magicpoint_path = "../magicpoint.pt"
magicpoint_model = torch.load(magicpoint_path, map_location=torch.device('cpu'))
homoadpatiter1_path = "../homoAdaptiter1.pt"
homoadpatiter1 = torch.load(homoadpatiter1_path, map_location=torch.device('cpu'))
homoadpatiter2_path = "../homoAdaptiter2.pt"
homoadpatiter2 = torch.load(homoadpatiter2_path, map_location=torch.device('cpu'))
image_dir = '../Dataset/MSCOCO/Train'
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
sample_list = np.random.randint(0, len(image_list), size=(3, ))
sampled_image = [image_list[i] for i in sample_list]
sampled_image = [image_processing(file).unsqueeze(0) for file in sampled_image]
image_tensor = torch.cat(sampled_image, dim=0)
model = SuperPointNetBatchNorm()
model.load_state_dict(magicpoint_model)  # result of magicpoint keypoints
model.train(mode=False)
with torch.no_grad():
    output = model(image_tensor.unsqueeze(1))
    keypoint_magicpoint = output['semi']
model.load_state_dict(homoadpatiter1)  # result of homographic adaptation 1 keypoints
model.train(mode=False)
with torch.no_grad():
    output = model(image_tensor.unsqueeze(1))
    keypoint_adapt1 = output['semi']
model.load_state_dict(homoadpatiter2)  # results of homographic adaptation 2 keypoints
model.train(False)
with torch.no_grad():
    output = model(image_tensor.unsqueeze(1))
    keypoint_adapt2 = output['semi']
fig, axes = plt.subplots(3, 3)
for i in range(len(sample_list)):
    points = detector_post_processing(keypoint_magicpoint[i, ...], conf_threshold=0.15, NMS_dist=1, limit_detection=1000)
    points = np.asarray(list(zip(points[1], points[0])), np.int)
    axes[i, 0].imshow(points_to_2D(points, 216, 320, img=sampled_image[i].numpy().squeeze()*255), cmap='gray')
    points = detector_post_processing(keypoint_adapt1[i, ...], conf_threshold=0.15, NMS_dist=1, limit_detection=1000)
    points = np.asarray(list(zip(points[1], points[0])), np.int)
    axes[i, 1].imshow(points_to_2D(points, 216, 320, img=sampled_image[i].numpy().squeeze()*255), cmap='gray')
    points = detector_post_processing(keypoint_adapt2[i, ...], conf_threshold=0.15, NMS_dist=1, limit_detection=1000)
    points = np.asarray(list(zip(points[1], points[0])), np.int)
    axes[i, 2].imshow(points_to_2D(points, 216, 320, img=sampled_image[i].numpy().squeeze()*255), cmap='gray')
plt.show()
print('Done')

