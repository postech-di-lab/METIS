import math
import torch
import numpy as np

def HitRatio(ranklist):
    return bool(1 in ranklist)    

def NDCG(ranklist):
    pos_position = np.where(ranklist == 1)[0]
    
    if len(pos_position) == 0:
        return 0
    else:
        pos = pos_position[0]
        return math.log(2) / math.log(pos+2)         


def return_perf(predictions, foreach):       
    topks = [2,5,10,20]
    
    # Initialize metrics
    hrs, ndcgs = {}, {}
    for tk in topks:
        if foreach == False:
            hrs[tk] = 0
            ndcgs[tk] = 0
        else:
            hrs[tk] = []
            ndcgs[tk] = []
            
    
    for row in predictions:         
        inst = row # [1 pos + N negs]
        labels = np.array([0] * (len(inst)-1) + [1])        
        
        # To set wrong if all predictions are the same,
        # move the positive item to the end of the list.
        inst[[0, -1]] = inst[[-1, 0]] 
        sortidx = inst.argsort()        
        
        sorted_labels = labels[sortidx]
        
        for tk in topks:
            topk_labels = sorted_labels[:tk]            
            
            if foreach == False:
                hrs[tk] += HitRatio(topk_labels)
                ndcgs[tk] += NDCG(topk_labels)
            else:
                hrs[tk].append(HitRatio(topk_labels))
                ndcgs[tk].append(NDCG(topk_labels))
                
    
    numinst = predictions.shape[0]
    
    if foreach == False:
        for tk in topks:
            hrs[tk] /= numinst
            ndcgs[tk] /= numinst
        
    return hrs, ndcgs
        
def _cal_ranking_measures(loader, model, opt, mode, isperf, foreach=False):
    predictions = np.array([])
    all_output, all_label = [], []
    all_uid, all_iid = [], []      
    
    for i, batch_data in enumerate(loader):        
        if opt.model_name in ['peris']:            
            user, all_items, label, times_item, neighbors, neighbors_time, neighbors_freqbin = batch_data        
                
            if isperf == False:                
                dist = model.predict_for_multiple_item(user, all_items, mode, times_item, neighbors, neighbors_time, neighbors_freqbin)                
            else:                 
                dist = model.predict_for_multiple_item_IS(user, all_items, mode, times_item, neighbors, neighbors_time, neighbors_freqbin)
                
            uid = user[0] if len(user) > 1 else user
        else:
            uid = batch_data[0]
            all_items = batch_data[1]
            dist = model.predict_for_multiple_item(batch_data)            
            
        all_output.append(dist)        
        all_uid.append(uid)
        all_iid.append(all_items)

    all_output = torch.cat(all_output).cpu().data.numpy()
    all_uid = torch.cat(all_uid).cpu().data.numpy()
    all_iid = torch.cat(all_iid).cpu().data.numpy()    
    
    if opt.model_name not in ['peris']:
        all_output = all_output.reshape(-1, 101) # 1 positive + 100 negative items

    if opt.model_name == 'peris':
        all_output = -1 * all_output # PERIS outputs scores (not distances)        

    hrs, ndcgs = return_perf(all_output, foreach)    
    
    return hrs, ndcgs

def cal_measures(loader, model, opt, mode=None, isperf=False):
    model.eval()    
    
    results = _cal_ranking_measures(loader, model, opt, mode, isperf)
    
    model.train()
    
    return results

def get_each_score(loader, model, opt, mode=None, isperf=False):
    model.eval()    
    
    score = _cal_ranking_measures(loader, model, opt, mode, isperf, foreach=True)
    
    model.train()
    
    return score



def _cal_logit(loader, model, opt, mode, isperf, foreach=False):
    predictions = np.array([])
    all_output, all_label = [], []
    all_uid, all_iid = [], []  
    
    for i, batch_data in enumerate(loader):
        if opt.model_name in ['peris']:
            user, all_items, label, times_item, neighbors, neighbors_time, neighbors_freqbin = batch_data        
                
            if isperf == False:
                dist = model.predict_for_multiple_item(user, all_items, mode, times_item, neighbors, neighbors_time, neighbors_freqbin)
            else:
                dist = model.predict_for_multiple_item_IS(user, all_items, mode, times_item, neighbors, neighbors_time, neighbors_freqbin)
                
            uid = user[0] if len(user) > 1 else user        
        else:
            uid = batch_data[0]
            all_items = batch_data[1]
            dist = model.predict_for_multiple_item(batch_data)    

            
        all_output.append(dist)
        all_uid.append(uid)
        all_iid.append(all_items)

    all_output = torch.cat(all_output).cpu().data.numpy()
    all_uid = torch.cat(all_uid).cpu().data.numpy()
    all_iid = torch.cat(all_iid).cpu().data.numpy()    
    
    if opt.model_name not in ['peris']: 
        all_output = all_output.reshape(-1, 101) 

    if opt.model_name == 'peris':
        all_output = -1 * all_output

    return all_output
    
def get_logit(loader, model, opt, mode=None, isperf=False):
    model.eval()    
    
    logit = _cal_logit(loader, model, opt, mode, isperf, foreach=True)
    
    model.train()
    
    return logit


def get_pis(loader, model, opt, mode=None, isperf=False):
    model.eval()    
    
    logit = _get_pis(loader, model, opt, mode, isperf, foreach=True)
    
    model.train()
    
    return logit

def _get_pis(loader, model, opt, mode, isperf, foreach=False):
    predictions = np.array([])
    all_output, all_label = [], []
    all_uid, all_iid = [], []  
    all_pis = []
    
    for i, batch_data in enumerate(loader):
        user, all_items, label, times_item, neighbors, neighbors_time, neighbors_freqbin = batch_data        

        pis = model.get_pis(user, all_items, mode, times_item, neighbors, neighbors_time, neighbors_freqbin)
            
        all_pis.append(pis)

    all_pis = torch.cat(all_pis).cpu().data.numpy()

    return all_pis