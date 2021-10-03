import os
import tqdm
import shutil

destination_dir = "../Dataset/MSCOCO/Validation_unlabelled"
label_dir = "../Dataset/Labels_MSCOCO/Validation"
image_dir = "../Dataset/MSCOCO/Validation"
labels = [file[:-4] for file in os.listdir(label_dir)]
images = [file[:-4] for file in os.listdir(image_dir)]
rats = []
for image in tqdm.tqdm(images):
    if image not in labels:
        rats.append(image)
for rat in rats:
    shutil.move(os.path.join(image_dir, rat + '.jpg'), os.path.join(destination_dir, rat + '.jpg'))
print(rat)
