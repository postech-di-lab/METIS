import os
import numpy as np
import pickle
import torch
import pandas as pd
import random
import csv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import subprocess
from .evaluation_all import *

current_file_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(os.path.dirname(current_file_dir))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_reg_loss(args, device, user, client_model, old_item_emb, old_user_emb=None):
    # item_emb, user_emb -> in self.device
    
    reg_fn = nn.MSELoss(reduction=args.reg_reduction)
    
    old_item_size = old_item_emb.weight.shape[0]
    
    reg_loss = 0
    if args.reg_d > 0:
        
        if old_user_emb is not None:
            old_user_size = old_user_emb.weight.data.shape[0]
            
            if user < old_user_size:
                old_user_emb_ = old_user_emb.weight.data[user].to(device)
                reg_loss += reg_fn(client_model.user_emb.weight.reshape(-1), old_user_emb_.reshape(-1))
        
        old_item_emb_ = old_item_emb.weight.data.to(device)
        reg_loss += reg_fn(client_model.item_emb.weight[:old_item_size], old_item_emb_)
        
    return reg_loss
            

def get_hyper_param(args):    
    hyper_param = {"seed": args.seed, "dim": args.dim, "lr": args.lr, "optim": args.optimizer, "weight_decay": args.weight_decay, 'patience': args.patience, "job_id": args.job_id}
    
    if args.backbone == 'fedmlp' or args.backbone == 'pfedrec':
        hyper_param.update({"lr_eta": args.lr_eta})
    
    if args.model == 'fcrec':    
        if args.client_cl and args.server_cl:
            hyper_param.update({'reg_client_cl': args.reg_client_cl, 'eps': args.eps, 'topN': args.topN, 'beta': args.beta, 'diff': args.diff, 'kd_user': args.kd_user, 'step': args.two_step_kd})
    
    
    return hyper_param


def save_baseblock_param(args, model):
    # save_dir = f"/home/jaehyunglim/F3CRec/base_model_param/{args.dataset}"
    # save_path = os.path.join(save_dir, f"{args.baseline}_{args.topN}_{args.standard}.pth")
    # os.makedirs(save_dir, exist_ok=True)
    
    save_dir = os.path.join(project_root, "base_model_param", args.dataset)
    save_path = os.path.join(save_dir, f"{args.baseline}_{args.topN}_{args.standard}.pth")

    os.makedirs(save_dir, exist_ok=True)

    torch.save(model, save_path)
 
    
def load_baseblock_param(args):
    #save_path =f"/home/jaehyunglim/F3CRec/base_model_param/{args.dataset}/{args.baseline}_{args.topN}_{args.standard}.pth"
    save_path = os.path.join(project_root, "base_model_param", args.dataset, f"{args.baseline}_{args.topN}_{args.standard}.pth")

    return torch.load(save_path)


def save_result_as_csv(args, model, RESULT, block_info, result_or_forget=None, client_cl=None, server_cl=None):
    if args.standard == 'NDCG':
        required_keys = ["LA_N@20", "RA_N@20", "HM_N@20"]
    elif args.standard == 'Recall':
        required_keys = ["LA_R@20", "RA_R@20", "HM_R@20"]
    
    # Check if all required keys are present and values are not None or NaN
    if not all(key in RESULT for key in required_keys) or any(RESULT[key] is None or np.isnan(RESULT[key]) for key in required_keys):
        return
    
    hyper_param = get_hyper_param(args)
    if args.model == 'fcrec':
        if args.client_cl and args.server_cl:
            save_model = 'fcrec'
        elif args.client_cl and not args.server_cl:
            save_model = 'cli_cl'
        elif not args.client_cl and args.server_cl:
            save_model = 'server_cl'
    else:
        save_model = args.model
        
    #dir_path = f"/home/jaehyunglim/F3CRec/fcrec_result/{args.dataset}/{args.backbone}/{save_model}"
    dir_path = os.path.join(project_root, "fcrec_result", args.dataset, args.backbone, str(save_model))


    if args.model == 'ablation':
        dir_path = dir_path + '/' + args.ablation
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    evaluation_metric = args.standard
    title = evaluation_metric
    csv_path = os.path.join(dir_path, title + ".csv")
    
    file_exists = os.path.isfile(csv_path)
    csv_head = ["block", "N@20", "R@20", "standard", "LA", "RA", "HM"] + list(hyper_param.keys())
    
    metric_type = "N@20" if args.standard == 'NDCG' else "R@20"
    row = [block_info] + [RESULT['N@20'], RESULT['R@20'], args.standard] + [RESULT[metric + '_' + metric_type] for metric in ["LA", "RA", "HM"]] + list(hyper_param.values())
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if not file_exists:
            writer.writerow(csv_head)  # Write header if the file does not exist
        writer.writerow(row)  # Write the data row


def get_num_user_item(total_blocks):

    user_item_info = {}

    block_user = set()
    block_item = set()

    for idx, block in enumerate(total_blocks):
        cur_user = set(block.user.values.tolist())
        cur_item = set(block.item.values.tolist())

        block_user = block_user.union(cur_user)
        block_item = block_item.union(cur_item)

        num_user = len(block_user)
        num_item = len(block_item)

        user_item_info[f"TASK_{idx}"] = {'num_user': num_user, 'num_item': num_item}

    return user_item_info



def make_rating_mat(dict, is_train, num_user, num_item):
    if is_train:
        rating_mat = {}
        for u,items in dict.items():
            u_dict = {i : 1 for i in items}
            rating_mat.update({u : u_dict})
            
        rating_mat_tensor = torch.zeros(num_user, num_item)
        
        for u, items in dict.items():
            rating_mat_tensor[u][items] = 1.0
            
        return rating_mat, rating_mat_tensor
        
    else: 
        rating_mat_tensor = torch.zeros(num_user, num_item)
        for u, items in dict.items(): 
            rating_mat_tensor[u][items] = 1.0

        return rating_mat_tensor
    

def make_interaction(dict):
    interactions = []
    for u, items in dict.items():
        for i in items:
            interactions.append((u, i, 1))
    interactions = list(set(interactions))
    return interactions


def get_rank_discrepancy_kd_loss(args, target_mat, pred_mat, last_topn_list, device):
    '''
        Adaptive Replay Memory & KD Loss
    '''
    bceloss = nn.BCELoss()
    
    topN = args.topN
    
    top_item_target = last_topn_list
    
    # check ranking
    sorted_indices = torch.argsort(pred_mat, descending=True) 
    rankings_pred = torch.empty_like(sorted_indices).to(device)
    rankings_pred[sorted_indices] = torch.arange(len(pred_mat)).to(device)
    ranks_pred_in_target_topN = rankings_pred[top_item_target] + 1
    
    rankings_target = torch.empty_like(sorted_indices).to(device)
    rankings_target[top_item_target] = torch.arange(len(top_item_target)).to(device)
    ranks_in_target_topN = rankings_target[top_item_target] + 1
    
    ranking_discrepancy = torch.sum(torch.abs(ranks_in_target_topN - ranks_pred_in_target_topN).clone().detach())

    # Consistency rate
    sample_rate = torch.exp(-args.eps * ranking_discrepancy)

        
    kd_idx = top_item_target.to('cpu')
    num_samples = int(len(kd_idx) * sample_rate) 
    
    if num_samples == 0:
        return 0
    
    # Adaptive Replay Memory
    indices = np.random.choice(len(kd_idx), num_samples, replace=False)
    kd_idx = kd_idx[indices].to(device)
    
    target = torch.sigmoid(target_mat[kd_idx])
    prediction = torch.sigmoid(pred_mat[kd_idx])
    
    # KD Loss for client-side continual learning
    kd_loss = bceloss(prediction, target)
    
    return kd_loss


def diff(old_emb, new_emb):
    '''
        get item-wise knowledge shift
    '''
    num_old_item, d = old_emb.size()
    num_new_item = len(new_emb)
    zero_delta = torch.zeros((num_new_item - num_old_item,1))
 
    delta = 1 + (1/torch.sqrt(torch.tensor(d))) * torch.sum((old_emb - new_emb[:len(old_emb)]) ** 2, dim=1)
    delta = torch.reciprocal(delta).reshape(-1, 1) 
    
    delta = torch.concat((delta, zero_delta), dim=0)
    return delta
    

def get_LA_RA(task, A_N, A_R):
    
    # LA = avg(a_{i,i]})
    # RA = avg(a_{k, i})
    LA_N = 0
    RA_N = 0
    
    LA_R = 0
    RA_R = 0
    
    
    for i in range(0, task+1):
        LA_N += A_N[i][i]
        LA_R += A_R[i][i]
        
        RA_N += A_N[task][i]
        RA_R += A_R[task][i]
    
    LA_N, LA_R, RA_N, RA_R = LA_N/(task+1), LA_R/(task+1), RA_N/(task+1), RA_R/(task+1)
    
    HM_N = (2 * LA_N * RA_N)/(LA_N + RA_N) if LA_N + RA_N != 0 else 0
    HM_R = (2 * LA_R * RA_R)/(LA_R + RA_R) if LA_R + RA_R != 0 else 0
    

    RESULT = {
              "LA_N@20": LA_N.item(),
              "RA_N@20": RA_N.item(),
              "LA_R@20": LA_R.item(),
              "RA_R@20": RA_R.item(),
              "HM_N@20": HM_N.item(),
              "HM_R@20": HM_R.item(),
              "N@20": A_N[task][task].item(),
              "R@20": A_R[task][task].item()
              }

    return RESULT


def get_device(args, threshold=None):
    if args.specific_gpu == 0:
        #free_gpus = get_free_gpus(threshold)
        # print(f"free gpu: {free_gpus}!!!!!")
        device = torch.device(f"cuda:0")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    
    return device
