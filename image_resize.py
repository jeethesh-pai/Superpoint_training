from cv2 import cv2
import os
import tqdm
import shutil


directory = "../pytorch-superpoint/datasets/MSCOCO"
train_files = [os.path.join(directory, 'Train', file) for file in os.listdir(directory + '/Train/')]
num_files = len(os.listdir(directory + '/Train/'))
count = 1
for i in tqdm.tqdm(range(num_files)):
    if count < num_files / 2:
        shutil.move(train_files[i], train_files[i].replace('Train', 'Train_1'))
    else:
        shutil.move(train_files[i], train_files[i].replace('Train', 'Train_2'))
    count += 1



