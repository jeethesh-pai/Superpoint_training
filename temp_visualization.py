import cv2
import os
import matplotlib.pyplot as plt


directory = "../Dataset/HPatches/i_ajuntament"
image_list = [os.path.join(directory, image) for image in os.listdir(directory)]
images = [cv2.imread(image) for image in image_list if image[-3:] == 'ppm']
fig, axes = plt.subplots(2, len(images) // 2)
count = 0
for i in range(2):
    for j in range(len(images)//2):
        axes[i, j].imshow(images[count])
        axes[i, j].set_title(f"{image_list[count][-5:]}")
        count += 1
plt.show()
