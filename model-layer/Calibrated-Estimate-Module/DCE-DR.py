# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import argparse
import time

from dataset import load_data, trainval_split
from models import *
from utils2 import ndcg_func, rating_mat_to_sample, binarize, shuffle, recall_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))

def train_and_eval(dataset_prep, train_args, model_args, gpu, cal, verbose):
    t0 = time.time()
    
    ## Dataset
    num_user, num_item, top_k_list, top_k_names, x_train, x_val, x_test, y_train, y_val, y_test = dataset_prep
    
    ## random
    rand_seed = np.random.randint(2023)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    ## Model & Training
    mf = MF_DR_JL_CE(num_user, num_item, batch_size=train_args['batch_size'], batch_size_prop = train_args['batch_size_prop'], num_experts=model_args['num_experts'], embedding_k=model_args['embedding_k'],gpu=gpu)
    mf.cuda(gpu)
    print("#parameters:",sum(p.numel() for p in mf.parameters()))
    mf._compute_IPS(x_train, lr=model_args['lr_prop'], lamb=model_args['lamb_prop'], verbose=verbose)
    if cal:
        mf._calibrate_IPS_G(x_val, x_test, num_epoch=model_args['epoch_propcal'], lr=model_args['lr_propcal'], lamb=model_args['lamb_propcal'], verbose=verbose, G=train_args['G_cal'])
    mf.fit(x_train, y_train, x_val, y_val, lr=model_args['lr'], lamb=model_args['lamb_pred'], gamma=train_args['gamma'], G=train_args['G'], lr_imp=model_args['lr_imp'], lamb_imp=model_args['lamb_imp'], lr_impcal=model_args['lr_impcal'], lamb_impcal=model_args['lamb_impcal'], iter_impcal=model_args['iter_impcal'], verbose=verbose, cal=cal)

    ## Test
    test_pred = mf.predict(x_test)
    
    mse_mf = mse_func(y_test, test_pred)
    mae_mf = mae_func(y_test, test_pred)
    
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    
    #precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    #f1 = 2 / (1 / np.mean(precisions[top_k_names[0]]) + 1 / np.mean(recalls[top_k_names[2]]))

    if verbose:
        print("***"*5 + "[DR-JL]" + "***"*5)
        print("[DR-JL] test mse:", mse_mf)
        print("[DR-JL] test mae:", mae_mf)
        print("[DR-JL] test auc:", auc)
        print("[DR-JL] {}:{:.6f}, {:.6f}".format("NDCG", np.mean(ndcgs[top_k_names[4]]), np.mean(ndcgs[top_k_names[5]])))
        print("[DR-JL] {}:{:.6f}, {:.6f}".format("RECALL", np.mean(recalls[top_k_names[2]]), np.mean(recalls[top_k_names[3]])))

        print("time:{:.4f}".format(time.time()-t0))
        print("***"*5 + "[DR-JL]" + "***"*5)
    
    return mse_mf, mae_mf, auc, np.mean(ndcgs[top_k_names[4]]), np.mean(ndcgs[top_k_names[5]]), np.mean(recalls[top_k_names[2]]), np.mean(recalls[top_k_names[3]])

## with optuna
def para(args):
    if args.dataset=="coat":
        # args.train_args = {"batch_size":128, "batch_size_prop":1024, 'gamma': 0.08304395254221075, 'G': 2, 'G_cal': 10}
        # args.model_args = {'epoch_propcal': 200, 'num_experts': 5, "embedding_k":16, 'lr': 0.02147727102922974, 'lr_imp': 0.09060546870227401, 'lr_prop': 0.05297996762336807, 'lamb_pred': 0.003998099521328913, 'lamb_imp': 0.1001242206281713, 'lamb_prop': 0.31404569078222366, 'lr_propcal': 0.06903361966383179, 'lamb_propcal': 0.0006413108267228355, 'lr_impcal': 0.018902059873783936, 'lamb_impcal': 0.0006640410463179381, 'iter_impcal': 10}
        args.train_args = {"batch_size":128, "batch_size_prop":1024, 'gamma': 0.09034303384487367, 'G': 2, 'G_cal': 10}
        args.model_args = {"embedding_k":16, 'lr': 0.02163101877954095, 'lr_imp': 0.09079457767848882, 'lr_prop': 0.05928551165349911, 'lamb_pred': 0.004906652556221904, 'lamb_imp': 0.34022488892823594, 'lamb_prop': 0.25210417432435117, 'epoch_propcal': 200, 'num_experts': 5, 'lr_propcal': 0.06356563724604007, 'lamb_propcal': 0.000727232313504342, 'lr_impcal': 0.0422362331082399, 'lamb_impcal': 0.0006098292234285432, 'iter_impcal': 10}
        
    elif args.dataset=="yahoo":
        args.train_args = {"batch_size":4096, "batch_size_prop":32764, 'gamma': 0.0619854138650803, 'G': 1, 'G_cal': 10}
        args.model_args = {"embedding_k":64, 'lr': 0.02524089813431658, 'lr_imp': 0.01516597673147271, 'lr_prop': 0.29187221004623864, 'lamb_pred': 0.0001130388086104561, 'lamb_imp': 0.006778525625477606, 'lamb_prop': 0.09943069478348952, 'epoch_propcal': 200, 'num_experts': 20, 'lr_propcal': 0.017337488565153117, 'lamb_propcal': 0.019138738308614497, 'lr_impcal': 0.0802891095159508, 'lamb_impcal': 0.010940003884439066, 'iter_impcal': 20}

    elif args.dataset=="kuai":
        #args.train_args = {"batch_size":4096, "batch_size_prop":32764, 'gamma': 0.06035920706527887, 'G': 5, 'G_cal': 50}
        #args.model_args = {"embedding_k":32, 'lr': 0.03184548322372834, 'lr_imp': 0.028534084073783005, 'lr_prop': 0.36793858235057914, 'lamb_pred': 0.0004952236549445727, 'lamb_imp': 0.007128686961410106, 'lamb_prop': 0.023982080034976365, 'epoch_propcal': 200, 'num_experts': 10, 'lr_propcal': 0.07463108709274435, 'lamb_propcal': 0.06700101539187704, 'lr_impcal': 0.09048898614156553, 'lamb_impcal': 0.019892690097600495, 'iter_impcal': 50}
        args.train_args = {"batch_size":4096, "batch_size_prop":32764, 'gamma': 0.058343929763571334, 'G': 1, 'G_cal': 10}
        args.model_args = {"embedding_k":64, 'lr': 0.01761227387015056, 'lr_imp': 0.019901453021484383, 'lr_prop': 0.36425697634689, 'lamb_pred': 0.00010699191524038388, 'lamb_imp': 0.006746464470239954, 'lamb_prop': 0.08729901882066586, 'epoch_propcal': 200, 'num_experts': 20, 'lr_propcal': 0.010413744498270544, 'lamb_propcal': 0.06105362358012219, 'lr_impcal': 0.08047787526965179, 'lamb_impcal': 0.021384435034677185, 'iter_impcal': 20}

    return args

def data_prep(dataset_name):
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")        
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]
        
        train_dic, x_train, y_train, val_dic, x_val, y_val = trainval_split(x_train, y_train, num_user, num_item, start_user=1)
        x_train, y_train = shuffle(x_train, y_train)

    elif dataset_name == "yahoo":
        x_train, y_train, x_test, y_test = load_data("yahoo")
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
        train_dic, x_train, y_train, val_dic, x_val, y_val = trainval_split(x_train, y_train, num_user, num_item, start_user=1)

        x_train, y_train = shuffle(x_train, y_train)

    elif dataset_name == "kuai":
        x_train, y_train, x_test, y_test = load_data("kuai")
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
        
        train_dic, x_train, y_train, val_dic, x_val, y_val = trainval_split(x_train, y_train, num_user, num_item, start_user=1)
        x_train, y_train = shuffle(x_train, y_train)
        
    if dataset_name == "kuai":
        y_train = binarize(y_train, 1)
        y_val = binarize(y_val, 1)
        y_test = binarize(y_test, 1)
        
    else:
        y_train = binarize(y_train)
        y_val = binarize(y_val)
        y_test = binarize(y_test)
    
    ## Metrics
    top_k_list = [5, 10]
    top_k_names = ("precision_5", "precision_10", "recall_5", "recall_10", "ndcg_5", "ndcg_10", "f1_5", "f1_10")

    if dataset_name == "kuai":
        top_k_list = [50, 51]
        top_k_names = ("precision_50", "precision_51", "recall_50", "recall_51", "ndcg_50", "ndcg_51", "f1_50", "f1_51")
    
    return num_user, num_item, top_k_list, top_k_names, x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int) ## gpu number
    parser.add_argument('--dataset', default='coat', type=str)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--iter_num', default=3, type=int)
    parser.add_argument('--cal', default=1, type=int)
    
    args = parser.parse_args()
    
    ## data prep
    dataset_prep = data_prep(args.dataset)
    
    args = para(args)
    print("args:",args.train_args, args.model_args)
    
    metrics = [] # mse, mae, auc, ndcg, recall
    iter_num = args.iter_num
    t0 = time.time()
    for i in range(iter_num):
        metric = train_and_eval(dataset_prep, args.train_args, args.model_args, args.cuda, args.cal, args.verbose)
        metrics.append(metric)
    metrics_mean = np.array(metrics).mean(axis=0)
    metrics_std = np.array(metrics).std(axis=0)
    # print("mse:{:.5f}+-{:.5f}, mae:{:.5f}+-{:.5f}, auc:{:.5f}+-{:.5f}, ndcg:{:.5f}+-{:.5f} / {:.5f}+-{:.5f}, recall:{:.5f}+-{:.5f} / {:.5f}+-{:.5f}:".format(metrics_mean[0], metrics_std[0], metrics_mean[1], metrics_std[1], metrics_mean[2], metrics_std[2], metrics_mean[3], metrics_std[3], metrics_mean[4], metrics_std[4], metrics_mean[5], metrics_std[5], metrics_mean[6], metrics_std[6]))
    print("mse:{:.5f}+-{:.5f}, auc:{:.5f}+-{:.5f}, ndcg:{:.5f}+-{:.5f} / {:.5f}+-{:.5f}:".format(metrics_mean[0], metrics_std[0], metrics_mean[2], metrics_std[2], metrics_mean[3], metrics_std[3], metrics_mean[4], metrics_std[4]))
    print("time: {:.4f}".format(time.time()-t0))