import torchio as tio
import numpy as np
import pandas as pd
import os
import csv
import torch
import glob
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_dilation
import csv
from pathlib import Path
from hmap_gener_v1 import create_hmap_v4
import random

def write_to_csv(filename, name, coords, shape):
    try:
        # 打开文件以追加模式，如果文件不存在则创建
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # 写入列表的内容
            for coord in coords:
                data_list = ["'"+name, coord[0], coord[1], coord[2], coord[3], coord[4], coord[5], shape]
                writer.writerow(data_list)
        
        # print(f"Data written to {filename} successfully.")
    except Exception as e:
        print(f"Error writing to {filename}: {str(e)}")


def read_imgcoord_fromcsv(name, part):
    #* 读取csv文件中的世界坐标
    # name = '01190830220138'
    imgcoord = pd.read_csv(f'/data/julia/data_lymph/anno/{part}_refine.csv')
    raw = imgcoord[imgcoord['name']=="'" + name]
    coords = []
    for i in range(len(raw)):
        x = raw.iloc[i, 1]
        y = raw.iloc[i, 2]
        z = raw.iloc[i, 3]
        width = raw.iloc[i, 4]
        height = raw.iloc[i, 5]
        depth = raw.iloc[i, 6]
        coords.append([x, y, z, width, height, depth]) # 这个是图像坐标系

    return coords




def generate_data(data_root_path, part, name):

    img_path = data_root_path.joinpath(part).joinpath(name)
    file_name = img_path.iterdir()
    img = tio.ScalarImage(os.path.join(img_path, file_name[0]))

    # * 窗宽窗位设置一下
    clamped = tio.Clamp(out_min=-160., out_max=240.)
    clamped_img = clamped(img)
    # clamped_img.save(f'/data/julia/data_lymph/{part}_processed/{name}_clamp.nii.gz')

    # * resample到（0.7， 0.7， 0.7）
    resample = tio.Resample(0.7)
    clamped_img = resample(clamped_img)
    # print(clamped_img.spacing)
    
    # * 归一化到 0-1 之间
    data_max = clamped_img.data.max()
    data_min = clamped_img.data.min()
    norm_data = (clamped_img.data - data_min) / (data_max - data_min)
    shape = clamped_img.shape
    data = np.array(norm_data.data.squeeze(0))
    np.save(f'{data_root_path}/{part}_npy/{name}_image.npy', data)

    # * 读取csv文件中的世界坐标
    worldcoord = pd.read_csv(f'{data_root_path}/lymph_csv_refine/CTA_thin_std_{part}_lymph_refine.csv')
    csv_filename = f'{data_root_path}/lymph_csv_refine/{part}_refine.csv'
    raw = worldcoord[worldcoord['image_path'].str.contains(name)]
    coords = []
    for i in range(len(raw)):
        x = raw.iloc[i, 2]
        y = raw.iloc[i, 3]
        z = raw.iloc[i, 4]
        width = raw.iloc[i, 5]
        height = raw.iloc[i, 6]
        depth = raw.iloc[i, 7]
        coords.append([x, y, z, width, height, depth]) # 这个是世界坐标系
    # print(f'the world coords is {coords}')

    # * 把世界坐标系转化为图像坐标系
    origin = img.origin
    # print(f'the origin is {origin}')
    img_coords = []
    for coord in coords:
        img_coord = (np.array(coord[0:3]) - np.array(origin) * np.array([-1., -1., 1.]) ) / np.array([0.7, 0.7, 0.7]) # img.spacing
        coord[3: 6] = coord[3: 6] / np.array([0.7, 0.7, 0.7])
        img_coords.append([img_coord[0], img_coord[1], img_coord[2], coord[3], coord[4], coord[5]])   #! xyzwhd
    # print(f'the image coord is {img_coords}')

    # 调用函数来写入数据
    write_to_csv(csv_filename, name, img_coords, shape)

    # * 开始生成并且保存这个hmap
    hmap = create_hmap_v4(img_coords, shape[1:])
    np.save(f'{data_root_path}/{part}_npy/{name}_hmap.npy', hmap)

    return data, hmap


def region_generate(data_root_path, part, name):
    # * 开始生成这个region
    region_path = f'{data_root_path}/{part}_mask/{name}/mediastinum_comp.nii.gz'
    region = tio.ScalarImage(region_path)
    resample = tio.Resample(0.7)
    resampled_region = resample(region)
    resampled_region = np.array(resampled_region.data.squeeze(0))
    np.save(f'{data_root_path}/{part}_npy/{name}_region.npy', resampled_region)

    return resampled_region



def create_mask(coordinates, shape, reduce=4, save=False, name=''):
    
    arr = np.zeros(tuple(np.array(shape) // reduce)) 
    for coord in coordinates:
        x, y, z = coord[0: 3]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[int(x)][int(y)][int(z)] = 1
    if save:
        np.save('/public_bme/data/xiongjl/det//npy_data//{}_mask.npy'.format(name), arr)
    
    return arr


def create_whd(coordinates, shape, reduce=4, save=False):
    
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for i in range(len(coordinates)):
        x, y, z, w, h, d = coordinates[i]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = w
        arr[1][int(x)][int(y)][int(z)] = h
        arr[2][int(x)][int(y)][int(z)] = d
    if save:
        np.save('array.npy', arr)
    
    return arr


def create_offset(coordinates, shape, reduce=4, save=False):
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for coord in coordinates:
        x, y, z = coord[0:3]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = x - int(x)
        arr[1][int(x)][int(y)][int(z)] = y - int(y)
        arr[2][int(x)][int(y)][int(z)] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr



def crop_data_region(part, name, image, hmap, center, crop_size, p=0.6, augmentatoin=False):

    crop_width, crop_height, crop_depth = crop_size
    origin_coords = read_imgcoord_fromcsv(name, part)

    width, height, depth = image.shape[:]
    # pad the image if it's smaller than the desired crop size
    pad_width = max(0, crop_width - width)
    pad_height = max(0, crop_height - height)
    pad_depth = max(0, crop_depth - depth)
    if pad_height > 0 or pad_width > 0 or pad_depth > 0:
        image = np.pad(image, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        width, height, depth = image.shape[:]

    random.seed(1)
    if random.random() < p:
        x_c, y_c, z_c = center
        x1 = x_c - crop_width/2
        y1 = y_c - crop_height/2
        z1 = z_c - crop_depth/2

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        z1 = int(max(0, z1))

        x1 = int(min(x1, width-crop_width))
        y1 = int(min(y1, height-crop_height))
        z1 = int(min(z1, depth-crop_depth))

        x2 = x1 + crop_width
        y2 = y1 + crop_height
        z2 = z1 + crop_depth

    else:
        x1 = random.randint(0, width - crop_width)
        x2 = x1 + crop_width
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        z1 = random.randint(0, depth - crop_depth)
        z2 = z1 + crop_depth
    
    cropped_image = image[x1:x2, y1:y2, z1:z2]
    hmap_crop = hmap[x1:x2, y1:y2, z1:z2]

    cropped_points = [(x-x1,y-y1,z-z1,w,h,d) for (x,y,z,w,h,d) in origin_coords if x1 <= x < x2 and y1 <= y < y2 and z1 <= z < z2]

    # if augmentatoin == True:
    #     if random.random() < 0.5:
    #         pass
    #     elif random.random() < 0.8:
    #         cropped_image, cropped_points = rotate_img(cropped_image, cropped_points, rotation_range=(-15, 15))
    #         cropped_points = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in origin_coords if 0 <= x <= cropped_image.shape[0] and 0 <= y <= cropped_image.shape[1] and 0 <= z <= cropped_image.shape[2]]
    #     else:
    #         cropped_image = add_noise(cropped_image)

    #* bulid the other label
    mask = create_mask(cropped_points, crop_size, reduce=1) # 0.0s no save is so fast
    whd = create_whd(coordinates=cropped_points, shape=crop_size, reduce=1)
    offset = create_offset(coordinates=cropped_points, shape=crop_size, reduce=1)

    hmap_crop = torch.from_numpy(hmap_crop)
    offset = torch.from_numpy(offset)
    mask = torch.from_numpy(mask)
    whd = torch.from_numpy(whd)
    
    dct = {}
    dct['hmap'] = hmap_crop
    dct['offset'] = offset
    dct['mask'] = mask
    dct['whd'] = whd
    dct['input'] = cropped_image
    dct['new_coords'] = cropped_points
    dct['name'] = name
    dct['origin_coords'] = origin_coords

    return dct






