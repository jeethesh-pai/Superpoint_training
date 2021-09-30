from cv2 import cv2
import os
import tqdm


directory = "../pytorch-superpoint/datasets/MSCOCO"
train_files = [os.path.join(directory, 'Train', file) for file in os.listdir(directory + '/Train/')]
val_files = [os.path.join(directory, 'Validation', file) for file in os.listdir(directory + '/Validation/')]
for file in tqdm.tqdm(train_files):
    image = cv2.imread(file)
    image = cv2.resize(image, (320, 216))
    cv2.imwrite(file, image)

