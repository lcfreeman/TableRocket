import os
import sys
sys.path.append('/data1/lcpang/lc/project_table/master_project/TableMASTER-mmocr-master/table_recognition/PubTabNet-master/src')
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import json
import glob
import json_lines
import time
import pickle
from metric import TEDS
from multiprocessing import Pool
from tqdm import tqdm

def list2filew(filename,lists):
    with open(filename,'w',encoding='utf-8')as fw:
        for lis in lists:
            fw.write(f'{lis}\n')
            
def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（Intersection over Union）。
    
    参数：
    box1：第一个边界框，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
    box2：第二个边界框，格式为 (x1, y1, x2, y2)。

    返回：
    IoU：IoU值，范围在0到1之间。
    """
    #print(box1,box2)
    # 提取坐标信息
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    #print(x1_1, y1_1, x2_1, y2_1)
    #print(x1_2, y1_2, x2_2, y2_2)
    # 计算交集的坐标
    x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    # 计算交集面积和并集面积
    intersection_area = x_intersection * y_intersection
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    if intersection_area == 0:
        iou = 0
    else:
        if union_area == 0:
            iou = 0
        else:
        # 计算IoU
            iou = intersection_area / union_area

    return iou

def non_max_suppression(boxes, scores, iou_threshold):
    """
    非极大值抑制（NMS）算法，用于去除重叠的边界框。

    参数：
    boxes：边界框列表，每个边界框是一个四元组 (x1, y1, x2, y2)，表示左上角和右下角坐标。
    scores：边界框对应的置信度列表，与边界框列表的顺序一致。
    iou_threshold：IoU（重叠度）阈值，用于判断边界框是否重叠。

    返回：
    selected_boxes：经过NMS后剩余的边界框。
    """
    selected_boxes = []
    
    # 按照置信度降序排列
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    while len(sorted_indices) > 0:
        max_score_index = sorted_indices[0]
        selected_boxes.append(boxes[max_score_index])
        
        # 计算与选定边界框重叠度
        current_box = boxes[max_score_index]
        remaining_indices = []
        for i in range(1, len(sorted_indices)):
            index = sorted_indices[i]
            iou = calculate_iou(current_box, boxes[index])
            if iou < iou_threshold:
                remaining_indices.append(index)
        
        sorted_indices = remaining_indices
    
    return selected_boxes


# 示例用法
#boxes = [[50, 50, 200, 200], [150, 100, 300, 250], [150, 150, 300, 300]]
#scores = [0.9, 0.85, 0.95]
#iou_threshold = 0.5
#
#selected_boxes = non_max_suppression(boxes, scores, iou_threshold)
#print("Selected Boxes:", selected_boxes)

cellbbox = './PTN/e12result_queryinst_ptnval9115.pkl'

with open(cellbbox, 'rb') as fp:
    predDict = pickle.load(fp)
    
    
for i in range(1,10):
    save_file = f'./PTN/e12_nms0.0{i}_result_queryinst_ptnval9115.pkl'
    #if i==10:
    #    continue
    save_dict = dict()
    iou_threshold = i/100

    for ptem in tqdm(predDict.items()):
        filename = ptem[0]
        raw_result = ptem[1]
        raw_bbox = [rb[:4] for rb in raw_result]
        scores = [rb[4] for rb in raw_result]
        nms_result = non_max_suppression(raw_bbox, scores,iou_threshold)
        save_dict.update({filename:nms_result})
        #pass
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f)






