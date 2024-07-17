import os
import mmcv
import torch
import mmdet
import json
import json_lines
import glob
import cv2 as cv
from tqdm import tqdm
from mmdet.apis import init_detector,inference_detector,show_result_pyplot

def list2filew(filename,lists):
    with open(filename,'w',encoding='utf-8')as fw:
        for lis in lists:
            fw.write(f'{lis}\n')
            
def filter_result(result,w,h):
    new_list=[]
    for i in result[0]:
        for j in i:
            if j[4]>0.3:
                new_j = j.tolist()
                for jj in range(4):
                    if new_j[jj]<0:
                        new_j[jj]=0
                if new_j[0]>w:
                    new_j[0]=w
                if new_j[2]>w:
                    new_j[2]=w    
                if new_j[1]>h:
                    new_j[1]=h
                if new_j[3]>h:
                    new_j[3]=h
                new_list.append(new_j)
    return new_list
   
def save_result(test_list, model, save_list):
    for item in tqdm(test_list):
        img = cv.imread(item)
        filename = item.split('/')[-1]
        w,h = img.shape[1],img.shape[0]
        result = inference_detector(model,img)
        format_result = filter_result(result,w,h)
        save_list.append(json.dumps({'filename':filename,'result':format_result},ensure_ascii=False))
    return save_list
    
if __name__ == "__main__":    
    
    config_file = '/data1/lcpang/lc/project_table/config/cellwcls_queryinst_small_PTN_train.py'
    checkpoint_file = '/data1/lcpang/lc/project_table/model/query_cell/PTN/epoch_12.pth'
    model=init_detector(config_file,checkpoint_file)
    
    save_list = []
    
    test_chunk_folder = '/data1/lcpang/lc/project_table/dataset/Pubtabnet/val/'
    
    
    test_list = glob.glob(test_chunk_folder+'*.png')
    save_list = save_result(test_list,model,save_list)
    
    list2filew('/data1/lcpang/lc/project_table/master_project/result/query_inst/test/queryinst_result_test_ptn.jsonl',save_list)    





















