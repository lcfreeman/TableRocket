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

    # 计算IoU
    iou = intersection_area / union_area

    return iou

def evaluate_image_detection(ground_truths, predictions, iou_threshold=0.5):
    """
    计算目标检测模型的性能指标，包括精确度、召回率、F1分数和平均精确度（mAP）。

    参数：
    ground_truth：真实的目标边界框列表，每个边界框的格式为 (x1, y1, x2, y2)。
    predictions：模型生成的目标边界框列表，每个边界框的格式为 (x1, y1, x2, y2, confidence)，
                confidence 表示模型对该边界框的置信度。
    iou_threshold：IoU阈值，用于确定什么被认为是正确检测到的目标。

    返回：
    precision：精确度
    recall：召回率
    f1：F1分数
    mAP：平均精确度
    """
    total_iou = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    num_predictions = len(predictions)
    num_ground_truths = len(ground_truths)
    average_precisions = []

    for gt_box in ground_truths:
        iou_max = -1
        for pred_box in predictions:
            iou = calculate_iou(gt_box[:4], pred_box[:4])
            if iou > iou_max:
                iou_max = iou
                #best_pred_box = pred_box
        #total_iou+=iou_max

        if iou_max >= iou_threshold:
            true_positives += 1
            #average_precisions.append(best_pred_box[4])
        else:
            false_negatives += 1

    false_positives = num_predictions - true_positives
    
    for prediction in predictions:
        iou_list = [calculate_iou(prediction, gt_box) for gt_box in ground_truths]
        total_iou += max(iou_list)  # 使用每个预测框与最匹配的ground truth回归框的IoU

    average_iou = total_iou / max(num_predictions, num_ground_truths)
    
    if true_positives == 0:
        precision = 0
        recall = 0
        f1 = 0
        mAP = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        #mAP = np.mean(average_precisions)

    return precision, recall, f1, average_iou# ,mAP
def htmlPostProcess(text):
    #text = '<html><body><table>' + text + '</table></body></html>'
    #return text
    text = '<html><body><table>' + text.replace('<thead>','').replace('</thead>','').replace('<tbody>','').replace('</tbody>','') + '</table></body></html>'
    return text

def evaluate_image_generation(teds, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess(context)
    htmlGtContext = htmlPostProcess(gt_context)
    # Acc Evaluate
    if htmlContext == htmlGtContext:
        flag = 1
    else:
        flag = 0
        
    # TEDS Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)
    
    #print("FILENAME : {}".format(file_name))
    #print("SCORE    : {}".format(score))
    return score, flag
    
def compute_metrics1(predDict, gtstrDict, gtbboxDict, init_teds):
    save_list = []
    for cnt, ptem in enumerate(tqdm(predDict.items())):
        filename = ptem[0]
        #pred_txt = ''.join(ptem[1]['text'].split(','))
        pred_stru_html = ''.join(ptem[1]['text'].split(',')).replace('eb','td')
        #pred_txt = pred_txt.replace('<thead>','').replace('</thead>','').replace('<tbody>','').replace('</tbody>','').replace('<eb>','<td>').replace('</eb>','</td>')
        
        pred_box = [[tmp[0]-tmp[2]/2,tmp[1]-tmp[3]/2,tmp[0]+tmp[2]/2,tmp[1]+tmp[3]/2] for tmp in ptem[1]['bbox'].tolist() if sum(tmp)>0]
        gt_stru_html = gtstrDict[filename]
        gt_box = gtbboxDict[filename]
        
        precision, recall, f1, average_iou = evaluate_image_detection(gt_box, pred_box, iou_threshold=0.5)
        tedstr, flag = evaluate_image_generation(init_teds, pred_stru_html, gt_stru_html)
        save_list.append(json.dumps({'filename':filename,'metrics':{'precision':precision, 'recall':recall, 'f1':f1, 'average_iou':average_iou,'tedstr':tedstr, 'flag':flag}},ensure_ascii=False))
        #save_list.append(json.dumps({'filename':filename,'metrics':{'tedstr':tedstr, 'flag':flag}},ensure_ascii=False))
        
    return save_list   

def compute_metrics(predDict, gtbboxDict):
    save_list = []
    for cnt, ptem in enumerate(tqdm(gtbboxDict.items())):
        filename = ptem[0]
        #pred_txt = ''.join(ptem[1]['text'].split(','))
        #pred_stru_html = ''.join(ptem[1]['text'].split(',')).replace('eb','td')
        #pred_txt = pred_txt.replace('<thead>','').replace('</thead>','').replace('<tbody>','').replace('</tbody>','').replace('<eb>','<td>').replace('</eb>','</td>')
        
        gt_box = ptem[1]
        #gt_stru_html = gtstrDict[filename]
        pred_box = predDict[filename]
        
        precision, recall, f1, average_iou = evaluate_image_detection(gt_box, pred_box, iou_threshold=0.5)
        #tedstr, flag = evaluate_image_generation(init_teds, pred_stru_html, gt_stru_html)
        save_list.append(json.dumps({'filename':filename,'metrics':{'precision':precision, 'recall':recall, 'f1':f1, 'average_iou':average_iou}},ensure_ascii=False))
        #save_list.append(json.dumps({'filename':filename,'metrics':{'tedstr':tedstr, 'flag':flag}},ensure_ascii=False))
        
    return save_list


if __name__ == "__main__":
    # Initialize TEDS object
    #init_teds = TEDS(n_jobs=1)
    #gtstrFile = '/data1/lcpang/lc/project_table/master_project/result/query_inst/convert/gt_PT_1M_test_wohead_struchtml.pkl'
    #with open(gtstrFile, 'rb') as fp1:
    #    gtstrDict = pickle.load(fp1)
    
    gtboxFile = '/data1/lcpang/lc/project_table/dataset/GridCellLabel/PTNcellbox_GT.pkl'
    with open(gtboxFile, 'rb') as fp2:
        gtboxDict = pickle.load(fp2)
    
    predFile = '/data1/lcpang/lc/project_table/master_project/result/query_inst/result/CellBox/PTN/e7_nms0.03_result_queryinst_ptnval9115.pkl'
    with open(predFile,'rb')as f:
        predDict = pickle.load(f)
        
    save_list = compute_metrics(predDict, gtboxDict)
    #save_list = compute_metrics(predDict, gtstrDict, gtboxDict, init_teds)
    
    list2filew('./PTN_socre_wohead.jsonl',save_list)    
    
    data = json_lines.reader(open('./PTN_socre_wohead.jsonl','r'))
    
    total_precision=0
    total_recall=0
    total_f1=0
    total_average_iou=0
    total_tedstr=0
    total_acc=0
    for stem in tqdm(data):
        precision=stem['metrics']['precision']
        recall=stem['metrics']['recall']
        f1=stem['metrics']['f1']
        average_iou=stem['metrics']['average_iou']
        #tedstr=stem['metrics']['tedstr']
        #acc=stem['metrics']['flag']
        total_precision+=precision
        total_recall+=recall
        total_f1+=f1
        total_average_iou+=average_iou
        #total_tedstr+=tedstr
        #total_acc+=acc
    average_precision=total_precision/8860
    average_recall=total_recall/8860
    average_f1=total_f1/8860
    average_average_iou=total_average_iou/8860
    #average_tedstr=total_tedstr/93737
    #average_acc=total_acc/93737
    with open('./PTN_average_socre_wohead.txt','w')as f:
        av_dict = json.dumps({'average_precision':average_precision, 'average_recall':average_recall, 
                              'average_f1':average_f1, 'average_iou':average_average_iou},ensure_ascii=False)
        #av_dict = json.dumps({'average_tedstr':average_tedstr,
        #                      'average_acc':average_acc},ensure_ascii=False)
        f.write(av_dict)
    f.close()
    
    