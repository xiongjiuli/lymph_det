import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter



def create_hmap_v4(coordinates, shape):
    arr = np.zeros(shape)
    for coords in coordinates:
        coord = [int(x) for x in coords[0:3]]
        whd = [int(x) for x in coords[3:6]]
        kernel = create_gaussian_kernel_v4(whd)
        arr = place_gaussian(arr, kernel, coord)

    return arr


def place_gaussian(arr, kernel, pos):
    x, y, z = pos
    kx, ky, kz = kernel.shape
    # 计算高斯核在数组中的位置
    x1, x2 = max(0, x-kx//2), min(arr.shape[0], x+kx//2+1)
    y1, y2 = max(0, y-ky//2), min(arr.shape[1], y+ky//2+1)
    z1, z2 = max(0, z-kz//2), min(arr.shape[2], z+kz//2+1)
    # 计算高斯核在自身中的位置
    kx1, kx2 = max(0, kx//2-x), min(kx, kx//2-x+arr.shape[0])
    ky1, ky2 = max(0, ky//2-y), min(ky, ky//2-y+arr.shape[1])
    kz1, kz2 = max(0, kz//2-z), min(kz, kz//2-z+arr.shape[2])
    # 将高斯核放置在指定位置
    arr[x1:x2,y1:y2,z1:z2] = np.maximum(arr[x1:x2,y1:y2,z1:z2], kernel[kx1:kx2,ky1:ky2,kz1:kz2])

    return arr



def create_gaussian_kernel_v4(whd):
    size_max = int(np.max(whd))
    size_min = int(np.min(whd))
    size_mid = int(sorted(whd)[1])

    array_large = create_gaussian_base(size_max, 0.5)
    array_small = create_gaussian_base(size_min, 0.5)
    array_midum = create_gaussian_base(size_mid, 0.5)

    combined_kernel = combine_gaussian_kernels(array_large, array_small, array_midum)

    return combined_kernel



def combine_gaussian_kernels(kernel_large, kernel_small, kernel_midum):
    center_large = np.array(kernel_large.shape) // 2
    small_shape = np.array(kernel_small.shape[0]) // 2
    midum_shape = np.array(kernel_midum.shape[0]) // 2

    kernel_large[center_large[0] - small_shape : center_large[0] + small_shape + 1, 
                 center_large[1] - small_shape : center_large[1] + small_shape + 1, 
                 center_large[2] - small_shape : center_large[2] + small_shape + 1, ] += kernel_small[:, :, :]
    
    kernel_large[center_large[0] - midum_shape : center_large[0] + midum_shape + 1, 
                 center_large[1] - midum_shape : center_large[1] + midum_shape + 1, 
                 center_large[2] - midum_shape : center_large[2] + midum_shape + 1, ] += kernel_midum[:, :, :]
    
    arr_min = kernel_large.min()
    arr_max = kernel_large.max()
    normalized_arr = (kernel_large - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间
    # print(f'in the combine_gaussian_kernels , the max is {normalized_arr.max()}, the min is {normalized_arr.min()}')
    return normalized_arr




def create_gaussian_base(size, threshold):

    if size <= 9:
        _size = 9
        half_dis = (_size + 1) / 2.
    else:
        _size = size
        if _size % 2 != 1:  # 如果size是偶数就变成奇数
            half_dis = _size / 2.
            _size = _size + 1
        else:
            half_dis = (_size + 1) / 2.

    if threshold == 0.5:
        sigma = np.sqrt(half_dis**2 / (2 * np.log(2)))
    elif threshold == 0.8:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(5) - np.log(4))))
    elif threshold == 0.3:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(10) - np.log(3))))
    else:
        print(f'when x = distance, the y wrong input, now the threshold is {threshold}')

    kernel = np.zeros((int(_size), int(_size), int(_size)))
    center = tuple(s // 2 for s in (int(_size), int(_size), int(_size)))
    kernel[center] = 1
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)

    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间
    # print(f'in the create_gaussian_base , the max is {normalized_arr.max()}, the min is {normalized_arr.min()}')
    return normalized_arr