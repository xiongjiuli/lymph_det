###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

# import _init_paths
import os
import matplotlib.pyplot as plt
# from BoundingBox import BoundingBox
# from BoundingBoxes import BoundingBoxes
# from Evaluator import *
# from utils import *
from plot.BoundingBox import BoundingBox
from plot.BoundingBoxes import BoundingBoxes
from plot.Evaluator import *
from plot.utils import *
import numpy as np


def get_gtboxes(folder_path, filename):
    result = []
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                data = line.strip().split()
                x1, y1, z1, x2, y2, z2 = map(float, data[1:7])
                result.append([x1, y1, z1, x2, y2, z2])
    return result


def get_predboxes(pred_file_name, filename, confidence):
    result = []
    folder_path = f'/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/{pred_file_name}'
    if filename.endswith('.txt'):   
        txt_path = os.path.join(folder_path, filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    confi = float(data[1])
                    if confi >= confidence:
                        x1, y1, z1, x2, y2, z2 = map(float, data[2:8])
                        result.append([x1, y1, z1, x2, y2, z2])
        else:
            result.append([0., 0., 0., 0., 0., 0.])
            print(f'in the {filename} no pred bbox!')
    return result



def iou(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    if iou < 0:
        iou = - iou
        print('the iou < 0, and i do the iou = - iou')
    # print(f'the iou is {iou}, the interArea is {interArea}, the union is {union}')
    assert iou >= 0
    return iou
    
# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[3]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[3]:
        return False  # boxA is left of boxB
    if boxA[2] > boxB[5]:
        return False  # boxA is left of boxB
    if boxB[2] > boxA[5]:
        return False  # boxA is left of boxB
    if boxA[4] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[4]:
        return False  # boxA is below boxB
    return True
def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1) * (zB - zA + 1)

def _getUnionAreas(boxA, boxB, interArea=None):
    # print(f'the boxa is {boxA}, the boxb is {boxB}')
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    # print(f'the areaa is {area_A}, the areab is {area_B}')
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
        # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the iou is {area_A + area_B - interArea}')
    return float(area_A + area_B - interArea)

def _getArea(box):
    return (box[3] - box[0] + 1) * (box[4] - box[1] + 1) * (box[5] - box[2] + 1)



def filter_boxes(gt, pred, iou_confi):
    result = []
    for box in gt:
        overlap = False
        for p_box in pred:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result


def get_fp(gt, pred, iou_confi):
    result = []
    for box in pred:
        overlap = False
        for p_box in gt:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result



def getBoundingBoxes(confi, gt_filename, det_filename):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    # print(f'currentPath is {currentPath}')
    folderGT = os.path.join(currentPath,'bbox_txt', gt_filename)
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            z = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            d = float(splitLine[6])
            iMagshape = splitLine[7]
            # print(f'in the gt box the imageshape is {iMagshape}')
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                z,
                w,
                h,
                d,
                CoordinatesType.Absolute, iMagshape,
                BBType.GroundTruth,
                format=BBFormat.XYZX2Y2Z2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, 'bbox_txt', det_filename)
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            if confidence < confi:
                continue
            else:
                x = float(splitLine[2])
                y = float(splitLine[3])
                z = float(splitLine[4])
                w = float(splitLine[5])
                h = float(splitLine[6])
                d = float(splitLine[7])
                iMagshape = splitLine[8]
                # print(f'in the pred box the imageshape is {iMagshape}')
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    z,
                    w,
                    h,
                    d,
                    CoordinatesType.Absolute, iMagshape,
                    BBType.Detected,
                    confidence,
                    format=BBFormat.XYZX2Y2Z2)
                allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes



def plot_froc(pred_file_name, train_data, iou_confi=0.01, suffix=''):
    # print(pred_file_name)
    pred_file_name = pred_file_name.split('/')[-2]
    # print(f'int rhe plot froc the pred file name is {pred_file_name}')
    # 计算这个 fROC 曲线
    false_positives_per_image = []
    recall = []
    for i in range(40, 105, 5):
        i = round(i * 0.01, 2)
        # 取出这个prediction boxes 和 gt boxes
        
        if train_data == True:
            folder_path = f'/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/train_groundtruths' # 指定文件夹路径
            gtall_nodes = 0
            predall_nodes = 0
            tp_nodes = 0
            fp_nodes = 0
            
            for filename in os.listdir(folder_path):
                ground_truth_boxes = get_gtboxes(folder_path, filename)  # 这个就是提取gt
                pred_bboxes = get_predboxes(pred_file_name, filename, confidence=i)  # 这个就是根据confi来提取大于这个confi的bbox
                label = f'{pred_file_name} {suffix}'
                predall_nodes += len(pred_bboxes)
                no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)  # 得到没有被预测出来的gt_box #!这个地方应该再加上iou的一些设置
                fp = get_fp(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)
                tp_nodes += (len(ground_truth_boxes) - len(no_predbox))
                fp_nodes += len(fp)
                gtall_nodes += len(ground_truth_boxes)

        else:
            folder_path = f'/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/groundtruths' # 指定文件夹路径
            gtall_nodes = 0
            tp_nodes = 0
            fp_nodes = 0
            predall_nodes = 0
            for filename in os.listdir(folder_path):
                ground_truth_boxes = get_gtboxes(folder_path, filename)
                pred_bboxes = get_predboxes(pred_file_name, filename, confidence=i)
                label = f'{pred_file_name} {suffix}'
                predall_nodes += len(pred_bboxes)
                no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)  # 得到没有被预测出来的gt_box
                fp = get_fp(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)
                tp_nodes += (len(ground_truth_boxes) - len(no_predbox))
                fp_nodes += len(fp)
                gtall_nodes += len(ground_truth_boxes)


        tPs = tp_nodes / gtall_nodes  # recall 
        recall.append(tPs)

        if train_data == True:
            fps = fp_nodes / len(os.listdir('/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/train_groundtruths'))
        else:
            fps = fp_nodes / len(os.listdir('/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/groundtruths'))
        false_positives_per_image.append(fps)

    print(f'false_positives_per_image is {false_positives_per_image}')
    print(f'recall is {recall}')

    plt.plot(false_positives_per_image, recall, marker='o', label=label)
    # plt.xscale('symlog')
    # 为每个点添加标签
    labels = [f'{round(i * 0.01, 2)}' for i in range(40, 105, 5)]
    for i, label in enumerate(labels):
        plt.annotate(label, xy=(false_positives_per_image[i], recall[i]), xytext=(false_positives_per_image[i] + 0.006, recall[i] - 0.08),
                    arrowprops=dict(facecolor='gray', edgecolor='gray', arrowstyle='-'))



def plot_ap(pred_file_name, train_data, confi, iou_confi, suffix=''):
    # print(f'pred_file_name is {pred_file_name}')
    pred_file_name = pred_file_name.split('/')[-2]


    # some xiugai in the plot




    # print(pred_file_name)
    # print(f'in the plot ap the perd file namw is {pred_file_name}')
    # Read txt files containing bounding boxes (ground truth and detections)
    if train_data == True:
        boundingboxes = getBoundingBoxes(confi=confi, gt_filename='train_groundtruths', det_filename=pred_file_name)
        annotation = f'{pred_file_name} {iou_confi}{suffix}confi:{confi}'
    else:
        boundingboxes = getBoundingBoxes(confi=confi, gt_filename='groundtruths', det_filename=pred_file_name)
        annotation = f'{pred_file_name} {iou_confi}{suffix}confi:{confi}'
    # Uncomment the line below to generate images based on the bounding boxes
    # createImages(dictGroundTruth, dictDetected)
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()
    ##############################################################
    # VOC PASCAL Metrics
    ##############################################################
    # Plot Precision x Recall curve
    ap_str = evaluator.PlotPrecisionRecallCurve(
        boundingboxes,
        annotation=annotation,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iou_confi,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=False,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,
        showGraphic=False)  # Plot the interpolated precision curve
    
    return ap_str



def plot(config, txt_paths, epoch, timesteamp):
    # txt_path is the whole path
    for path in txt_paths:
        if 'train-' in path:
            txt_path_training = path
            # print('trianing=======================')
        else:
            txt_path_testing = path
            # print('testing=======================')
    plt.figure()
    iou_confi = config['iou_confi']
    ap_confi = config['ap_confi']
    # print(f'the txt_path_testing is {txt_path_testing}')
    ap_str_testing_01 = plot_ap(pred_file_name=txt_path_testing, train_data=False, confi=ap_confi, iou_confi=0.01, suffix='')
    # ap_str_testing_10 = plot_ap(pred_file_name=txt_path_testing, train_data=False, confi=ap_confi, iou_confi=0.1, suffix='')
    ap_str_testing_50 = plot_ap(pred_file_name=txt_path_testing, train_data=False, confi=ap_confi, iou_confi=0.5, suffix='')
    # ap_str_testing_75 = plot_ap(pred_file_name=txt_path_testing, train_data=False, confi=ap_confi, iou_confi=0.75, suffix='')

    ap_str_training_01 = plot_ap(pred_file_name=txt_path_training, train_data=True, confi=ap_confi, iou_confi=0.01, suffix='')
    # ap_str_training_10 = plot_ap(pred_file_name=txt_path_training, train_data=True, confi=ap_confi, iou_confi=0.1, suffix='')
    ap_str_training_50 = plot_ap(pred_file_name=txt_path_training, train_data=True, confi=ap_confi, iou_confi=0.5, suffix='')
    # ap_str_training_75 = plot_ap(pred_file_name=txt_path_training, train_data=True, confi=ap_confi, iou_confi=0.75, suffix='')

    plt.grid(True)
    plt.savefig(f"{config['image_save_path']}png_img/PR Curve {config['model_name']} iou {iou_confi}-thres {ap_confi}-{epoch}{timesteamp}.png")
    plt.close()

    plt.figure()
    plot_froc(pred_file_name=txt_path_testing, train_data=False, iou_confi=iou_confi, suffix=f'{iou_confi}')
    plot_froc(pred_file_name=txt_path_training, train_data=True, iou_confi=iou_confi, suffix=f'{iou_confi}')
    plt.grid()
    plt.legend()
    plt.xlabel('False Positives Per Image')
    plt.ylabel('True Positive Fraction (recall)')
    plt.title('fROC Curve')
    plt.savefig(f"{config['image_save_path']}png_img/fROC Curve {config['model_name']} iou{iou_confi}-{epoch}{timesteamp}.png")
    plt.close()

    results = {}
    results['AP_IoU_0.01'] = ap_str_testing_01
    # results['AP_IoU_0.10'] = ap_str_testing_10
    results['AP_IoU_0.50'] = ap_str_testing_50
    # results['AP_IoU_0.75'] = ap_str_testing_75
    results['AP_IoU_0.01_train'] = ap_str_training_01
    # results['AP_IoU_0.10_train'] = ap_str_training_10
    results['AP_IoU_0.50_train'] = ap_str_training_50
    # results['AP_IoU_0.75_train'] = ap_str_training_75

    return results


# if __name__ == '__main__':

#     config = {}
#     config['iou_confi'] = 0.1
#     config['ap_confi'] = 0.35
#     config['image_save_path'] = '/public_bme/data/xiongjl/lymph_det/'
#     config['model_name'] = 'swin3d'
#     txt_paths = ['/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/swin3d_50_295628/']
#     epoch = 50
#     timesteamp = 295628
#     # print(f'out the plot the txt_paths is {txt_paths}')
#     plot(config, txt_paths, epoch, timesteamp)



















    # plt.figure()
    # iou_confi = 0.05
    # ap_confi = 0.45
    # train_data = True

    # model_names = [

        

    #     'train-swin_crop160_hmapv4-1485',
    #     # 'train-swin_crop160_hmapv4-620',
    #     # 'swin_crop160_hmapv4-1485_9_0.1',
    #     # 'swin_crop160_hmapv4-1485_11_0.1',
    #     'train-swincsacade_crop160_hmapv4_best-65_7_0.15',
    #     # 'train-unetr_crop160_hmapv4-360',
    #     # 'unetr_crop160_hmapv4_best-160_9_0.1',
    #     # 'unetr_crop160_hmapv4_best-160_11_0.1',
    #     #train- 'res101_crop256_hmapv4_best-220',
    #     #train- 'res101_crop256_hmapv4-525',
    #     #train- 'res101_crop256_hmapv4-995',
    #     #train- 'train-swin_crop160_hmapv4-620',
    #     #train- 'swin_crop160_hmapv5_best-395', 
    #     # 'train-swin_crop160_hmapv5-600',
    #     # 'train-swin_crop160_hmapv5-860',
        
    # ]

    # for names in model_names:
    #     plot_ap(pred_file_name=names, train_data=train_data, confi=ap_confi, iou_confi=iou_confi, suffix='')

    # plt.grid(True)
    # # plt.show()
    # plt.savefig(f"/public_bme/data/xiongjl/uii/png_img/PR Curve iou {iou_confi}-thres {ap_confi}.png")

    
    # fig, axes = plt.subplots()
    # for name in model_names:
    #     plot_froc(pred_file_name=name, train_data=train_data, iou_confi=iou_confi, suffix='')


    # # x_ticks = [0, 1/8, 1/4, 1/2, 1, 2, 4, 8] # 生成x轴的刻度值
    # # y_ticks = np.arange(0, 1.1, 0.1) # 生成y轴的刻度值

    # # # 设置x轴和y轴的刻度
    # # plt.xticks(x_ticks, [str(val) for val in x_ticks])
    # # plt.yticks(y_ticks)
    # # 显示网格线
    # plt.grid()
    # plt.legend()
    # plt.xlabel('False Positives Per Image')
    # plt.ylabel('True Positive Fraction (recall)')
    # plt.title('fROC Curve')
    # # 保存图像为PNG格式文件
    # plt.savefig(f"/public_bme/data/xiongjl/uii/png_img/fROC Curve iou{iou_confi}.png")
    # # plt.show()
