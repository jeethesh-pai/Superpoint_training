import os
import numpy as np
import shutil

file_path = "../pytorch-superpoint/datasets/MSCOCO"
file_list = os.listdir(file_path)
# uncomment below line to make directory inside MSCOCO dataset
# os.mkdir(file_path + '/Train')
# os.mkdir(file_path + '/Validation')
sampler = np.random.choice(range(len(file_list)), 2783, replace=False)
validation_files = [file_path + '/Validation/' + file_list[i] for i in sampler]
source_files = [file_path + '/' + i for i in file_list]
for i in range(len(sampler)):
    shutil.move(file_list[sampler[i]], validation_files[i])
