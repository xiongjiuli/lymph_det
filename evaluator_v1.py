import csv
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import torchio as tio
from IPython import embed
import csv
import torch.nn.functional as F
import os
from tqdm import tqdm
import shutil
import random
from time import time
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#* the test is for window crop for a image


class DetectionEvaluator:
    def __init__(self, config) -> None:
        self._config = config
        self._patch_size = self._config['patch_size']
        self._overlap = self._config['overlap']
        # self._train_data = self._config['train_data']
        self._confidence = self._config['confidence']

    def __call__(self, model, number_epoch):
        test_names = []
        txt_paths = []
        file_paths = [self._config['test_training_mode_name_path'],
                      self._config['test_testing_mode_name_path']] # '/public_bme/data/xiongjl/uii/csv_files/part_testing_trainingnames.csv'

        for file_path in file_paths:
            test_names = read_names_from_csv(file_path)
            if 'training' in file_path:
                part = 'training'
                det_file = 'train-'
            else:
                part = 'testing'
                det_file = ''

            model.eval()
            scale = [1., 1., 1.]
            step = [pahs - ovlap for pahs, ovlap in zip(self._patch_size, self._overlap)]
            no_pred_dia = []
            for name in tqdm(test_names):
                data_root_path = Path(self._config['lymph_nodes_data_path'])
                image_path = data_root_path.joinpath(f'{part}_npy').joinpath(f'{name}_image.npy')
                image_data = np.load(image_path)
                shape = image_data.shape
                image_patches = sliding_window_3d_volume_padded(image_data, patch_size=self._patch_size, stride=step) 
                #* the image_patch is a list consist of the all patch of a whole image
                #* each element in the list is a dict consist of start point and tensor(input)
                label_xyzwhd = name2coord(self._config, part, name)
                # print(f'========================{name}========================')

                whole_hmap = np.zeros(image_data.shape())
                whole_whd = np.zeros(np.hstack(((3), image_data.shape())))
                whole_offset = np.zeros(np.hstack(((3), image_data.shape())))
                for image_patch in image_patches:
                    with torch.no_grad():
                        image_input = image_patch['image'].unsqueeze(0)
                        point = image_patch['point'][1:]
                        order = image_patch['point'][0]
                        image_input = image_input.cuda()
                
                        pred_hmap, pred_whd, pred_offset = model(image_input)

                        whole_hmap = place_small_image_in_large_image(whole_hmap, pred_hmap, point)
                        whole_whd = place_small_image_in_large_image(whole_whd, pred_whd, point)
                        whole_offset = place_small_image_in_large_image(whole_offset, pred_offset, point)
                        # pred_bbox = decode_bbox(pred_hmap, pred_whd, pred_offset, scale, self._confidence, reduce=1., cuda=True, point=point)
                        # pred_bboxes.append(pred_bbox)

                pred_bboxes = decode_bbox(self._config, whole_hmap, whole_whd, whole_offset, scale, self._confidence, reduce=1., cuda=True, point=point)

                ground_truth_boxes = centerwhd_2nodes(label_xyzwhd, point=(0, 0, 0))
            
                # do the nms
                pred_bboxes = normal_list(pred_bboxes)
                # pred_bboxes = non_overlapping_boxes(pred_bboxes)

                time_nms = time()
                print('nmsing.........')
                pred_bboxes = nms_(pred_bboxes, thres=self._config['nms_threshold'])
                print(f'nms time is {time() - time_nms}')
                # print(f'the gt bbox is {ground_truth_boxes}')
                # print(f'the pred bbox is {pred_bboxes}')
                # draw_boxes_on_nii(part, name, ground_truth_boxes, pred_bboxes)
                # no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes)
                # no_pred_dia.extend(no_predbox)

                # * 生成这个seg的mask
                # selected_box = select_box(pred_bboxes, 0.25)
                # create_boxmask(ground_truth_boxes, selected_box, image_shape=shape, name=name)
                for bbox in pred_bboxes:
                    hmap_score, x1, y1, z1, x2, y2, z2 = bbox
                    txt_path = f"{self._config['lymph_nodes_data_path']}/txts/{det_file}{self._config['model_name']}_{number_epoch}/"
                    txt_paths.append(txt_path)
                    if not os.path.exists(txt_path):
                        os.makedirs(txt_path)
                    with open(f"{txt_path}/{name}.txt", 'a') as f:
                        f.write(f'nodule {hmap_score} {x1} {y1} {z1} {x2} {y2} {z2} {shape}\n')
            # print(f'the no pred dia is {no_pred_dia}') 
            # print(f'the confidence is {confidence}')
            # print(f'the model path is {model_path}')
            # print(f'the train_data is {train_data}') 
        return txt_paths
    


def pool_nms(heat, kernel):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(config, pred_hms, pred_whds, pred_offsets, scale, confidence, reduce, point, cuda):

    pred_hms    = pool_nms(pred_hms, kernel = config['decode_box_kernel_size'])
    heat_map    = pred_hms[0, :, :, :].cpu()
    pred_whd    = pred_whds[0, :, :, :].cpu()
    pred_offset = pred_offsets[0, :, :, :].cpu()

    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(1).bool()
    mask.cuda()
    # mask[0, 50, 60, 30] = 1
    indices = np.argwhere(mask == 1)

    xyzwhds = []
    hmap_scores = []
    for i in range(indices.shape[1]):
        coord = indices[1 :, i]
        x = coord[0].cpu()
        y = coord[1].cpu()
        z = coord[2].cpu()
        
        offset_x = pred_offset[0, x, y, z]
        offset_y = pred_offset[1, x, y, z]
        offset_z = pred_offset[2, x, y, z]
        # embed()
        hmap_score = heat_map[0, x, y, z]
        # print(f'--x y z -- : {x}, {y}, {z}')
        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]

        center = ((x + offset_x) * reduce, (y + offset_y) * reduce, (z + offset_z) * reduce)
        center = [a / b for a, b in zip(center, scale)]

        xyzwhds.append([center[0], center[1], center[2], w, h, d])
        hmap_scores.append(hmap_score)
    predicted_boxes = centerwhd_2nodes(xyzwhds, point=point, hmap_scores=hmap_scores)
    # print(f'predicted_boxes is : {predicted_boxes}')

    return predicted_boxes


def pad_image(image, target_size):
    # 计算每个维度需要填充的数量
    padding = [(0, max(0, target_size - size)) for size in image.shape]
    # 使用pad函数进行填充
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # 返回填充后的图像
    return padded_image


def sliding_window_3d_volume_padded(arr, patch_size, stride, padding_value=0):
    """
    This function takes a 3D numpy array representing a 3D volume and returns a 4D array of patches extracted using a sliding window approach.
    The input array is padded to ensure that its dimensions are divisible by the patch size.
    :param arr: 3D numpy array representing a 3D volume
    :param patch_size: size of the cubic patches to be extracted
    :param stride: stride of the sliding window
    :param padding_value: value to use for padding
    :return: 4D numpy array of shape (num_patches, patch_size, patch_size, patch_size)
    """
    # regular the shape
    if len(arr.shape) != 3:
        arr = arr.squeeze(0)

    patch_size_x = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_z = patch_size[2]

    stride_x = stride[0]
    stride_y = stride[1]
    stride_z = stride[2]

    # Compute the padding size for each dimension
    pad_size_x = (patch_size_x - (arr.shape[0] % patch_size_x)) % patch_size_x
    pad_size_y = (patch_size_y - (arr.shape[1] % patch_size_y)) % patch_size_y
    pad_size_z = (patch_size_z - (arr.shape[2] % patch_size_z)) % patch_size_z

    # Pad the array
    arr_padded = np.pad(arr, ((0, pad_size_x), (0, pad_size_y), (0, pad_size_z)), mode='constant', constant_values=padding_value)

    # Extract patches using a sliding window approach
    patches = []
    order = 0
    for i in range(0, arr_padded.shape[0] - patch_size_x + 1, stride_x):
        for j in range(0, arr_padded.shape[1] - patch_size_y + 1, stride_y):
            for k in range(0, arr_padded.shape[2] - patch_size_z + 1, stride_z):
                patch = arr_padded[i:i + patch_size_x, j:j + patch_size_y, k:k + patch_size_z]
                if isinstance(patch, np.ndarray):
                    patch = torch.from_numpy(patch).unsqueeze(0)
                else:
                    patch = patch.unsqueeze(0)
                start_point = torch.tensor([order, i, j, k])
                add = {'image': patch, 'point': start_point}
                patches.append(add)
                order += 1
    # return np.array(patches)
    return patches



def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


def select_box(predbox, p):
    selected_box = []
    for box in predbox:
        i = box[0]
        if i >= p:
            selected_box.append(box)
    return selected_box



def nms_(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    x1 = [det[1] for det in dets]
    y1 = [det[2] for det in dets]
    z1 = [det[3] for det in dets]
    x2 = [det[4] for det in dets]
    y2 = [det[5] for det in dets]
    z2 = [det[6] for det in dets]
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) * (z2[i] - z1[i]) for i in range(len(x1))]
    scores = [det[0] for det in dets]
    order = order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    print(f'in the nms, the len of dets is {len(dets)}')
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = [max(x1[i], x1[j]) for j in order[1:]]
        xx2 = [min(x2[i], x2[j]) for j in order[1:]]
        yy1 = [max(y1[i], y1[j]) for j in order[1:]]
        yy2 = [min(y2[i], y2[j]) for j in order[1:]]
        zz1 = [max(z1[i], z1[j]) for j in order[1:]]
        zz2 = [min(z2[i], z2[j]) for j in order[1:]]

        w = [max(xx2[i] - xx1[i], 0.0) for i in range(len(xx1))]
        h = [max(yy2[i] - yy1[i], 0.0) for i in range(len(yy1))]
        d = [max(zz2[i] - zz1[i], 0.0) for i in range(len(zz1))]

        inters = [w[i] * h[i] * d[i] for i in range(len(w))]
        unis = [areas[i] + areas[j] - inters[k] for k, j in enumerate(order[1:])]
        ious = [inters[i] / unis[i] for i in range(len(inters))]

        inds = [i for i, val in enumerate(ious) if val <= thres]
         # return the rest boxxes whose iou<=thres

        order = [order[i + 1] for i in inds]

            # inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3
    result = [dets[i] for i in keep]
    print(f'after the nms, the len of result is {len(result)}')
    return result



def non_overlapping_boxes(boxes):
    non_overlapping = []
    for i, box1 in enumerate(boxes):
        overlapping = False
        for j, box2 in enumerate(non_overlapping):
            if boxes_overlap(box1, box2):
                overlapping = True
                if box_area(box1) > box_area(box2):
                    non_overlapping.remove(box2)
                    non_overlapping.append(box1)
                break
        if not overlapping:
            non_overlapping.append(box1)
    return non_overlapping


def boxes_overlap(box1, box2):
    x1, y1, z1, x2, y2, z2 = [np.float16(x) for x in box1[1:]]
    a1, b1, c1, a2, b2, c2 = [np.float16(x) for x in box2[1:]]
    return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1 or z2 < c1 or c2 < z1)


def box_area(box):
    _, x1, y1, z1, x2, y2, z2 = box
    return (x2 - x1) * (y2 - y1) * (z2 - z1)



def name2coord(config, part, mhd_name):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyzwhd = []
    # csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_file_dir = f"/public_bme/data/xiongjl/uii/csv_files/{part}_refine.csv"
    with open(csv_file_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            
            if row[0] == "'" + mhd_name:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                w = float(row[4]) 
                h = float(row[5]) 
                d = float(row[6]) 
                xyzwhd.append((x, y, z, w, h, d))
    return xyzwhd




import numpy as np

def place_small_image_in_large_image(large_image, small_image, start_coords):
    
    if (start_coords[0] < 0 or start_coords[1] < 0 or start_coords[2] < 0 or
            start_coords[0] + small_image.shape[-3] > large_image.shape[-3] or
            start_coords[1] + small_image.shape[-2] > large_image.shape[-2] or
            start_coords[2] + small_image.shape[-1] > large_image.shape[-1]):
        raise ValueError("小图像的起始坐标超出大图像范围")
    
    # 获取小图像的坐标范围
    x_start, y_start, z_start = start_coords
    x_end = x_start + small_image.shape[-3]
    y_end = y_start + small_image.shape[-2]
    z_end = z_start + small_image.shape[-1]
    
    # 将小图像放入大图像中，选择最大值
    if len(large_image.shape) == 3:
        large_image[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )

    elif len(large_image.shape) == 4:
        large_image[:, x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[:, x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )
    else:
        print(f'large image shape should be 3 or 4, but now is {len(large_image.shape)}')
    return large_image


def centerwhd_2nodes(xyzwhds, point, hmap_scores=None):
    if hmap_scores != None:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd, hmap_score in zip(xyzwhds, hmap_scores):

            x, y, z, length, width, height = xyzwhd
            x1 = x - length/2.0
            y1 = y - width/2.0
            z1 = z - height/2.0
            x2 = x + length/2.0
            y2 = y + width/2.0
            z2 = z + height/2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([hmap_score, x1, y1, z1, x2, y2, z2])
        return result
    
    else:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd in xyzwhds:

            x, y, z, length, width, height = xyzwhd
            x1 = x - length / 2.0
            y1 = y - width / 2.0
            z1 = z - height / 2.0
            x2 = x + length / 2.0
            y2 = y + width / 2.0
            z2 = z + height / 2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([x1, y1, z1, x2, y2, z2])

        return result
    

def normal_list(list):
    new_list = []
    for lit in list:
        if lit == []:
            continue
        else:
            for l in lit:
                new_list.append(l)
    return new_list