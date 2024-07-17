import sys
import os

sys.path.append('/data/lcpang/lc/project_table/table_analysis/master_project/TableMASTER-mmocr-master/table_recognition/PubTabNet-master/src')


import json
import time
import pickle
from tqdm import tqdm
from metric import TEDS
from multiprocessing import Pool

def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text

def htmlPostProcess1(text):
   #text = '<html><body><table>' + text + '</table></body></html>'
   #return text
    text = '<html><body>' + text.replace('<thead>','').replace('</thead>','').replace('<tbody>','').replace('</tbody>','') + '</body></html>'
    return text

def singleEvaluation(teds, file_name, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess(context)
    htmlGtContext = htmlPostProcess1(gt_context)
    # Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)

    #print("FILENAME : {}".format(file_name))
    #print("SCORE    : {}".format(score))
    return score

def singleEvaluation1(teds, file_name, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess1(context)
    htmlGtContext = htmlPostProcess1(gt_context)
    # Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)

    #print("FILENAME : {}".format(file_name))
    #print("SCORE    : {}".format(score))
    return score

if __name__ == "__main__":

    thr = 0.03
    t_start = time.time()
    pool = Pool(64)
    start_time = time.time()
    #gtJsonFile = '/data1/lcpang/lc/project_table/dataset/Pubtabnet/ptn_val_strußhtml_gt_wohead.pkl'
    gtJsonFile = '/data/lcpang/lc/project_table/57result/TEDS/FTN/ftn_test10571_struhtml_gt_wohead.pkl'
    predFile = f'/data/lcpang/lc/project_table/57result/TEDS/FTN/e7_nms{thr}_query_struhtml_result.pkl'

    # Initialize TEDS object
    teds = TEDS(n_jobs=1)

    with open(predFile, 'rb') as f:
        predDict = pickle.load(f)

    with open(gtJsonFile, 'rb') as f:
        gtValDict = pickle.load(f)

    #assert len(predDict) == len(gtValDict)

    # # cut 10 to debug
    # file_names = [p for p in predDict.keys()][:10]
    # cut_predDict = dict()
    # for file_name in file_names:
    #     cut_predDict.setdefault(file_name, predDict[file_name])
    # predDict = cut_predDict

    scores = []
    caches = dict()
    save_file = dict()
    for idx, (file_name, context) in enumerate(tqdm(predDict.items())):
        # loading
        # file_name = os.path.basename(file_path)
        gt_context = gtValDict[file_name]
        # print(file_name)
        score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, context, gt_context,))
        scores.append(score)
        #save_file.update({file_name:score})
        tmp = {'score':score, 'gt':gt_context, 'pred':context}
        caches.setdefault(file_name, tmp)

    pool.close()
    pool.join() # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()

    # get score from scores
    cal_scores = []
    for score in scores:
        cal_scores.append(score.get())
    avg_score = sum(cal_scores)/10571
    print('AVG TEDS score: {}'.format(avg_score))
    print('TEDS cost time: {}s'.format(time.time()-start_time))
    
    print("Save cache for analysis.")
    
    save_folder = '/data/lcpang/lc/project_table/57result/TEDS/score'
    for file_name in caches.keys():
        info = caches[file_name]
        if info['score']._value < 1.0:
            save_file.update({file_name:info['score']._value})
    #with open(os.path.join(save_folder, 'content2html2_stedsscorefilename.pkl'), 'wb')as f:
        #pickle.dump(save_file, f)
    with open(os.path.join(save_folder, f'e7FTN{thr}_stedsscore.txt'), 'w')as f:
        f.write(str(avg_score))
        
    for file_name in caches.keys():
        info = caches[file_name]
        #if info['score']._value < 1.0:
        save_file.update({file_name:info['score']._value})
            
    with open(os.path.join(save_folder, f'e7FTN{thr}_stedsscorefilename.pkl'), 'wb')as f:
        pickle.dump(save_file, f)
    #for file_name in caches.keys():
    #    info = caches[file_name]
    #    if info['score']._value < 1.0:
    #        f = open(os.path.join(save_folder, file_name.replace('.png', '.txt')), 'w')
    #        f.write(file_name+'\n'+'\n')
    #        f.write('Score:'+'\n')
    #        f.write(str(info['score']._value)+'\n'+'\n')
    #        f.write('Pred:'+'\n')
    #        f.write(info['pred']+'\n'+'\n')
    #        f.write('Gt:' + '\n')
    #        f.write(info['gt']+'\n'+'\n')





