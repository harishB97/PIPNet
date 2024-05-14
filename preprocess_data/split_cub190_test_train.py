import os
import shutil
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import argparse

def alter_name_for_cub_phylogeny(name):
    return 'cub_' + name[:3] + '_' + name[4:]

parser = argparse.ArgumentParser()
parser.add_argument('--path',
                    type=str,
                    default='data/CUB_200_2011/',
                    help='Path to CUB-200-2011 dataset')
args = parser.parse_args()
path = args.path

images_path = "images_cub190/"

time_start = time.time()

path_images = os.path.join(path,'images.txt')
path_split = os.path.join(path,'train_test_split.txt')
train_save_path = os.path.join(path,'dataset_cub190/train_crop/')
test_save_path = os.path.join(path,'dataset_cub190/test_crop/')
bbox_path = os.path.join(path, 'bounding_boxes.txt')

images = []
with open(path_images,'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))

bboxes = dict()
with open(bbox_path, 'r') as bf:
    for line in bf:
        id, x, y, w, h = tuple(map(float, line.split(' ')))
        bboxes[int(id)]=(x, y, w, h)

num = len(images)
folders_skipped = set()
for k in tqdm(range(num), total=num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    file_name = alter_name_for_cub_phylogeny(file_name)

    if not os.path.exists(os.path.join(path, images_path, file_name)):
        folders_skipped.add(file_name)
        continue

    if int(split[k][0][-1]) == 1:
        
        if not os.path.isdir(train_save_path + file_name):
            os.makedirs(os.path.join(train_save_path, file_name))
        # img = Image.open(os.path.join(os.path.join(path, images_path),images[k][0].split(' ')[1])).convert('RGB')
        img = Image.open(os.path.join(path, images_path, file_name, images[k][0].split(' ')[1].split('/')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x+w, y+h))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
    else:
        if not os.path.isdir(test_save_path + file_name):
            os.makedirs(os.path.join(test_save_path,file_name))
        img = Image.open(os.path.join(path, images_path, file_name, images[k][0].split(' ')[1].split('/')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x+w, y+h))
        cropped_img.save(os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))

print('Done cropping')
print(f'{len(folders_skipped)} folders skipped')

train_save_path = os.path.join(path,'dataset_cub190/train/')
test_save_path = os.path.join(path,'dataset_cub190/test/')

folders_skipped = set()
num = len(images)
for k in tqdm(range(num), total=num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    file_name = alter_name_for_cub_phylogeny(file_name)

    if not os.path.exists(os.path.join(path, images_path, file_name)):
        folders_skipped.add(file_name)
        continue

    if int(split[k][0][-1]) == 1:
        if not os.path.isdir(train_save_path + file_name):
            os.makedirs(os.path.join(train_save_path,file_name))
        shutil.copy(os.path.join(path, images_path, file_name, images[k][0].split(' ')[1].split('/')[1]),
                    os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
    else:
        if not os.path.isdir(test_save_path + file_name):
            os.makedirs(os.path.join(test_save_path,file_name))
        shutil.copy(os.path.join(path, images_path, file_name, images[k][0].split(' ')[1].split('/')[1]),
                    os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
time_end = time.time()
print(f'{len(folders_skipped)} folders skipped')
print('CUB190, %s!' % (time_end - time_start))
