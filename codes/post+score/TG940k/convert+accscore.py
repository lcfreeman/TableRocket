import os
import cv2 as cv
import numpy as np
import copy
import random
import time
import glob
import pickle
import json
import json_lines
from tqdm import tqdm
import multiprocessing
import shutil
import copy
import xml.etree.ElementTree as ET
from IPython.core.display import display, HTML

def list2filew(filename,lists):
    with open(filename,'w',encoding='utf-8')as fw:
        for lis in lists:
            fw.write(f'{lis}\n')

def group_numbers_by_threshold(numbers, threshold):
    groups = []
    current_group = [numbers[0]]

    for i in range(1, len(numbers)):
        diff = numbers[i] - current_group[0]

        if diff < threshold:
            current_group.append(numbers[i])
        else:
            groups.append(current_group)
            current_group = [numbers[i]]

    groups.append(current_group)  # 添加最后一个组
    #print(groups)
    # 计算每个组的平均值并取整
    group_averages = [round(sum(group) / len(group)) for group in groups]
    
    return group_averages

# 示例
#numbers = [1, 3, 5, 8, 10, 12, 15, 16, 17, 18,19]
#threshold = 2
#
#result = group_numbers_by_threshold(numbers, threshold)
#print(result)
#

def recover_table_layout(bounding_boxes):
    # 收集边界框的坐标信息
    x_coords = []
    y_coords = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box[:4]
        x_coords.append(x1)
        x_coords.append(x2)
        y_coords.append(y1)
        y_coords.append(y2)

    # 将 x 和 y 坐标排序并去重
    #x_coords = sorted(list(set(x_coords)))
    #y_coords = sorted(list(set(y_coords)))
    
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    # 根据最小wh阈值将 x 和 y 坐标分组
    x_threshold = min([i[2] - i[0] for i in bounding_boxes])
    y_threshold = min([i[3] - i[1] for i in bounding_boxes])

    # 初始化虚拟线
    virtual_horizontal_lines = []
    virtual_vertical_lines = []

    virtual_vertical_lines = group_numbers_by_threshold(x_coords,x_threshold)
    virtual_horizontal_lines = group_numbers_by_threshold(y_coords,y_threshold)
   
    # 返回虚拟竖直线和虚拟水平线
    return virtual_vertical_lines, virtual_horizontal_lines

# 示例代码不变
#bounding_boxes = [
#    (100, 100, 200, 200),  # 框1
#    (250, 100, 350, 200),  # 框2
#    (100, 200, 200, 300),  # 框3
#    (299, 200, 399, 300),  # 框4
#    (100, 300, 200, 400),  # 框5
#    (250, 300, 350, 400)   # 框6
#]
#
## 恢复表格辅助线
#horizontal_lines, vertical_lines = recover_table_layout(bounding_boxes)
#
## 打印恢复的虚拟水平线和虚拟竖直线
#print("虚拟水平线:")
#print(horizontal_lines)
#print("虚拟竖直线:")
#print(vertical_lines)
#

def get_cell_coord(vertical_lines,horizontal_lines):
    """
    input:  虚拟水平线,虚拟竖直线
    output: 虚拟最小粒度单元格列表(未合并)
    """
    
    total_cell = []
    cell_coord = []
    
    for i in horizontal_lines:
        tr_cell = []
        for j in vertical_lines:
            tr_cell.append((j,i))
        total_cell.append(tr_cell)
    
    for rdx, rtem in enumerate(total_cell):
        #print(idx,item)
        if rdx < len(total_cell)-1:
            temp = zip(total_cell[rdx][:-1],total_cell[rdx+1][1:])
            r_cell = list(temp)
            cell_coord.append(r_cell)
    return cell_coord 


def get_cell_coord_tag(cell_coord,pre_list):
    """
    input:  虚拟最小粒度单元格列表(未合并),预测bbox列表
    output: 最小粒度单元格bbox, 最小粒度单元格自定tag(r,c)代表包含当前单元格的情况下合并r行,c列属于[1:∞],1等价于不合并.
    """
    
    
    t = cell_coord 

    for r in range(len(t)):
        for i in range(len(t[r])):
        #print(t[r][i])
            c_p = ((t[r][i][0][0]+t[r][i][1][0])/2,(t[r][i][0][1]+t[r][i][1][1])/2)
            for j in pre_list:
                if c_p[0]>j[0] and c_p[0]<j[2] and c_p[1]>j[1] and c_p[1]<j[3]:
                    t[r][i]=((j[0],j[1]),(j[2],j[3]))
    cell_coord = t
    transpose = [list(row) for row in zip(*cell_coord)]

    tag_list_X = []
    for r in range(len(transpose)):
        temp = []
        for i in range(len(transpose[r])):
            #print(cell_coord[r].count(cell_coord[r][i]))
            temp.append((transpose[r].count(transpose[r][i]),0))
        tag_list_X.append(temp)
    trans_tag_list_X = [list(row) for row in zip(*tag_list_X)]
    
    tag_list_Y = []
    for r in range(len(cell_coord)):
        temp = []
        for i in range(len(cell_coord[r])):
            #print(cell_coord[r].count(cell_coord[r][i]))
            temp.append((0,cell_coord[r].count(cell_coord[r][i])))
        tag_list_Y.append(temp)
        
    nptag = list(np.add(trans_tag_list_X,tag_list_Y))
    tag = [i.tolist() for i in nptag]
                    
    return cell_coord, tag

def merge_spanning(t):
    """
    input: 最小粒度单元格bbox二维列表
    output: 合并单元格后bbox二维列表
    """
    temp = []
    one_temp = []
    for r in t:
        rtemp = []
        for i in r:
            if i not in one_temp:
                one_temp.append(i)
                rtemp.append(i)
        if len(rtemp):
            temp.append(rtemp)
    return temp

def merge_tag(tag):
    """
    input: 最小粒度单元格tag
    output: 合并单元格后tag
    """
    #print(tag)
    for i in range(len(tag)):
        for j in range(len(tag[i])):
            #print(tag[i][j])
            if tag[i][j][1]>1:
                x_d = tag[i][j][1]-1
                #print(x_d)
                #print(i,j)
                for k in range(x_d):
                    tag[i][j+1+k]=(0,0)
            if tag[i][j][0]>1:
                y_d = tag[i][j][0]-1
                for k in range(y_d):
                    tag[i+1+k][j]=(0,0)
            if tag[i][j][0]>1 and tag[i][j][1]>1:
                #print(x_d,y_d)
                for k in range(x_d):
                    for kk in range(y_d):
                        tag[i+1+kk][j+1+k]=(0,0)
                        
    new_tag_list = []
    for i in range(len(tag)):
        temp = []
        for j in range(len(tag[i])):
            #print(sum(j))
            if sum(tag[i][j]):
                temp.append(tag[i][j])
        if len(temp):        
            new_tag_list.append(temp)
    #print(new_tag_list)
    return new_tag_list

def get_html_list(merge_cell, str_coord, new_tag_list):
    """
    input: 合并后单元格二维list, 原始标签bbox_list, 合并后单元格二维tag
    output: 表格标准HTML_list
    """
    html_str = []
    #flag = 0
    #head_row = get_header(merge_cell, str_coord)
    #if head_row+1 == len(merge_cell):
    #    flag = 1
    #html_str.append("<thead>")
    for i in range(len(new_tag_list)):
        html_str.append("<tr>")
        for j in range(len(new_tag_list[i])):
            #print(tag[i][j])
            if new_tag_list[i][j][1]>1 or new_tag_list[i][j][0]>1:
                html_str.append("<td")
                cspan = new_tag_list[i][j][1]
                rspan = new_tag_list[i][j][0]
                if cspan>1 and rspan>1:
                    html_str.append(f" colspan=\"{cspan}\" rowspan=\"{rspan}\"")
                else:
                    if cspan>1:
                        html_str.append(f" colspan=\"{cspan}\"")
                    if rspan>1:
                        html_str.append(f" rowspan=\"{rspan}\"")
                html_str.append(">")
                
                
            else:
                html_str.append("<td>")    
            html_str.append("</td>") 
        html_str.append("</tr>")
        #if i==head_row:
        #    html_str.append("</thead>")
        #    html_str.append("<tbody>")
    #html_str.append("</tbody>")
    
    return html_str#, flag


def get_ocr_text(las_cell,ocr_item):
    """
    input: merged table cellbboxlist, ocr_data(bbox+text _format)
    output: ordered_text_list, to matching stru_html
    """
    ocr_list = []
    for i in range(len(las_cell)):
        for j in range(len(las_cell[i])):
            rbox = las_cell[i][j]
            #print(las_cell[i][j])
            tem_txt = []
            for otem in ocr_item:
                ocr_box = otem['bbox']
                ocr_txt = otem['text']
                c_x, c_y = (ocr_box[0]+ocr_box[2])/2, (ocr_box[1]+ocr_box[3])/2
                if c_x>rbox[0][0] and c_x<rbox[1][0] and c_y>rbox[0][1] and c_y<rbox[1][1]:
                    tem_txt.append(ocr_txt)
            
            ocr_list.append(' '.join(tem_txt))
        
    return ocr_list

def get_table_html(stru_html,ocr_list):
   
    """
    input: table structure html list, ocr text list
    output: table html sequence
    """
    
    rev_orc_txt_list = ocr_list[::-1]
    insert_list = []
    test = copy.deepcopy(stru_html)
    for idx, item in enumerate(stru_html):
        if item =='</td>':
           # print(idx,item)
            insert_list.append(idx)
    for idx,item in enumerate(insert_list[::-1]):
        test.insert(item,rev_orc_txt_list[idx])
        
    table_html = ''.join(test)
    return table_html

def get_structure_pred_result(pre_list):
    """
        input: 单元格预测bboxlist,ocrdata(bbox,text)
        output: table structure html ,table html sequence
    """
    virtual_vertical_lines, virtual_horizontal_lines = recover_table_layout(pre_list)
    cell_coord_t = get_cell_coord(virtual_vertical_lines,virtual_horizontal_lines)
    cell_coord,tag = get_cell_coord_tag(cell_coord_t,pre_list)
    las_cell = merge_spanning(cell_coord)
    las_tag = merge_tag(tag)
    stru_html_list = get_html_list(las_cell,pre_list,las_tag)
    stru_html = ''.join(stru_html_list)

    return stru_html

def get_table_pred_result(pre_list, stru_html_list, ocr_item):
    """
        input: 单元格预测bboxlist,ocrdata(bbox,text)
        output: table structure html ,table html sequence
    """
    virtual_vertical_lines, virtual_horizontal_lines = recover_table_layout(pre_list)
    cell_coord_t = get_cell_coord(virtual_vertical_lines,virtual_horizontal_lines)
    cell_coord,tag = get_cell_coord_tag(cell_coord_t,pre_list)
    las_cell = merge_spanning(cell_coord)
    las_tag = merge_tag(tag)
    #stru_html_list = get_html_list(las_cell,pre_list,las_tag)
    ocr_list = get_ocr_text(las_cell,ocr_item)
    table_html = get_table_html(stru_html_list,ocr_list)
    #stru_html = ''.join(stru_html_list)

    return table_html

def get_format_pred_result(pre_list,ocr_item):
    
    virtual_vertical_lines, virtual_horizontal_lines = recover_table_layout(pre_list)
    cell_coord_t = get_cell_coord(virtual_vertical_lines,virtual_horizontal_lines)
    cell_coord,tag = get_cell_coord_tag(cell_coord_t,pre_list)
    las_cell = merge_spanning(cell_coord)
    las_tag = merge_tag(tag)
    stru_html_list = get_html_list(las_cell,pre_list,las_tag)
    structure_html = ''.join(stru_html_list)
    ocr_list = get_ocr_text(las_cell,ocr_item)
    table_html = get_table_html(stru_html_list,ocr_list)
    
    return structure_html, table_html

def evaluate_image_generation(context1, gt_context1):
    # save problem log
    # save_folder = ''

    # html format process
    #htmlContext1 = htmlPostProcess(context1)
    #htmlContext2 = htmlPostProcess(context2)
    #htmlGtContext1 = htmlPostProcess(gt_context1)
    #htmlGtContext2 = htmlPostProcess(gt_context2)
    # Acc Evaluate
    if context1== gt_context1:
        flag = 1
    else:
        flag = 0
        
    # TEDS Evaluate
    #score1 = teds.evaluate(htmlContext1, htmlGtContext1)
    #score2 = teds.evaluate(htmlContext2, htmlGtContext2)
    #print("FILENAME : {}".format(file_name))
    #print("SCORE    : {}".format(score))
    return flag


if __name__ == "__main__": 
    
    stru_gt_path = '/data1/lcpang/lc/project_table/master_project/result/query_inst/convert/gt_PT_1M_test_wohead_struchtml.pkl'
    save_folder = f'/data1/lcpang/lc/project_table/master_project/result/query_inst/convert/result/'
    for ithr in range(11,20):
        thr = ithr/100
        queryinst_result = f'/data1/lcpang/lc/project_table/master_project/result/query_inst/result/nms{thr}_result_queryinst(PT-1M_cellbbox)_PT-1Mtest93737.pkl'
        with open(queryinst_result, 'rb') as fp2:
            predDict = pickle.load(fp2)
       #### get structure result
        
        save_dict = dict()
        err_list = []
        save_file = save_folder+f'e9_nms{thr}_query_html_result.pkl'
        for qtem in tqdm(predDict.items()):
            filename = qtem[0]
            pre_list = qtem[1]
            try:
                structure_html = get_structure_pred_result(pre_list)
                save_dict.update({filename:structure_html})
                #save_list.append(json.dumps({'filename':filename,'html':{'structure_html':structure_html,'table_html':table_html}},ensure_ascii=False))
            #break
            except Exception as e:
                err_list.append(filename)
        with open(save_file, 'wb') as f:
            pickle.dump(save_dict, f)  
        list2filew(save_folder+f'e9_nms{thr}_err_list.txt',err_list)
        
        stru_pred_path = f'/data1/lcpang/lc/project_table/master_project/result/query_inst/convert/result/e9_nms{thr}_query_html_result.pkl'
        
        
        with open(stru_gt_path, 'rb') as fp1:
            stru_gt_dict = pickle.load(fp1)
        with open(stru_pred_path, 'rb') as fp2:
            wohead_stru_html_dict = pickle.load(fp2)
        
       
        save_list = []
        for ptem in tqdm(wohead_stru_html_dict.items()):
            filename = ptem[0]
            pred_stru_html = ptem[1]
            #pred_wohead_table_html = ptem['html']['table_html']
            gt_stru_html = stru_gt_dict[filename]        
            #gt_wohead_table_html = wohead_table_html_dict[filename]
            flag = evaluate_image_generation(pred_stru_html, gt_stru_html)
            save_list.append(json.dumps({'filename':filename,'metrics':{'flag':flag}},ensure_ascii=False))
        list2filew(save_folder+f'e9{thr}acc.jsonl',save_list)    
        
        data=json_lines.reader(open(save_folder+f'e9{thr}acc.jsonl','r'))
        
        total_tedtab=0
        total_tedstr=0
        total_acc=0
        for stem in tqdm(data):
            #tedtab=stem['metrics']['tedtab']
            #tedstr=stem['metrics']['tedstr']
            acc=stem['metrics']['flag']
            #total_tedtab+=tedtab
            #total_tedstr+=tedstr
            total_acc+=acc
        #average_tedtab=total_tedtab/93737
        #average_tedstr=total_tedstr/93737
        average_acc=total_acc/93737
        with open(save_folder+f'e9{thr}_average_acc.txt','w')as f:
            av_dict = json.dumps({'average_acc':average_acc},ensure_ascii=False)
            f.write(av_dict)
        f.close()