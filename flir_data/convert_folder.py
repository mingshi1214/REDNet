import os
import sys
import cv2
from tqdm import tqdm

thermal_folders = [el for el in os.listdir(os.curdir) if "images_thermal" in el or "video_thermal" in el]
print("iterating over the following folders")
print(thermal_folders)

print("outputs will be in root dir of the respective folder type under the folder small with the exact same name as the original")
for folder in thermal_folders:
    parent = os.path.join(folder, 'data')
    images = os.listdir(os.path.join(folder, 'data'))
    os.makedirs(os.path.join(folder,'small'), exist_ok=True)
    for image in tqdm(images):
        img = cv2.imread(os.path.join(parent,image), 1)
        new_dim = (160, 120)
        nuimg = cv2.resize(img, new_dim, interpolation=cv2.INTER_CUBIC)
        nupath = os.path.join(parent, image).replace('data', 'small')
        cv2.imwrite(nupath, nuimg)

print("Done")
