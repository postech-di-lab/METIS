# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time
import random
from utils2 import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import torch

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(abs(x-y))

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, gpu=0, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.gpu = gpu

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:, 1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        
    def forward_logit(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:, 1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    
    def get_emb(self, x):
        user_idx = torch.LongTensor(x[:, 0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:, 1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        return U_emb, V_emb
    
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

    def predict_logit(self, x):
        pred = self.forward_logit(x)
        return pred.detach().cpu()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method."""
    def __init__(self, num_users, num_items, embedding_k=4, gpu=0, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.gpu = gpu
        self.xent_func = torch.nn.BCELoss()
        
        # for calibration
        self.a = None
        self.b = None

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:,1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        

    def forward_logit(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:, 1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.linear_1(z_emb)

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)
        
    def forward_cal(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:,1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.a * self.linear_1(z_emb) + self.b)

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)

    def get_emb(self, x):
        user_idx = torch.LongTensor(x[:, 0]).cuda(self.gpu)
        item_idx = torch.LongTensor(x[:, 1]).cuda(self.gpu)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        return U_emb, V_emb
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()
    
    def predict_logit(self, x):
        pred = self.forward_logit(x)
        return pred.detach().cpu()
    
class MF_DR_JL_CE(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, num_experts, embedding_k=4, gpu=0, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.gpu = gpu
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, gpu=gpu, *args, **kwargs)
        self.imputation_model = MF_BaseModel(num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, gpu=gpu)
        self.propensity_model = NCF_BaseModel(num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, gpu=gpu, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        
        ## for calibration
        self.num_experts = num_experts
        self.prop_selection_net = nn.Sequential(nn.Linear(self.embedding_k, num_experts), nn.Softmax(dim=1))
        self.imp_selection_net = nn.Sequential(nn.Linear(self.embedding_k, num_experts), nn.Softmax(dim=1))

        self.a_prop = torch.FloatTensor([1 for i in range(num_experts)]).cuda(self.gpu).requires_grad_()
        self.b_prop = torch.FloatTensor([-1 for i in range(num_experts)]).cuda(self.gpu).requires_grad_()
        self.a_imp = torch.FloatTensor([1 for i in range(num_experts)]).cuda(self.gpu).requires_grad_()
        self.b_imp = torch.FloatTensor([-1 for i in range(num_experts)]).cuda(self.gpu).requires_grad_()
        
        self.sm = nn.Softmax(dim = 1)
     
    def calibration_experts(self, x, T, mode='prop'):
        # get emb
        if mode == 'prop':
            u_emb, _ = self.propensity_model.get_emb(x)
            logit = self.propensity_model.forward_logit(x)
        else:
            u_emb, _ = self.imputation_model.get_emb(x)
            logit = self.imputation_model.forward_logit(x)
        #print(u_emb.device, logit.device)
        
        # get selection dist (Gumbel softmax)
        if mode == 'prop':
            selection_dist = self.prop_selection_net(u_emb) # (batch_size, num_experts)
        else:
            selection_dist = self.imp_selection_net(u_emb)
        #print(selection_dist.device)
        
        #g = torch.distributions.Gumbel(torch.tensor([0.0]).cuda(self.gpu), torch.tensor([1.0]).cuda(self.gpu)).sample(selection_dist.size()).squeeze() #.cuda(self.gpu)
        g = torch.distributions.Gumbel(torch.tensor(0.0).cuda(self.gpu), torch.tensor(1.0).cuda(self.gpu)).sample(selection_dist.size()) #.cuda(self.gpu)
        #print(g.size(), selection_dist.size())
        eps = torch.tensor(1e-10).cuda(self.gpu) # for numerical stability
        selection_dist = selection_dist + eps
        selection_dist = self.sm((selection_dist.log() + g) / T) # (batch_size, num_experts) (row sum to 1)
        #print(g.device, eps.device, T.device, selection_dist.device)
        
        # calibration experts
        logits = torch.unsqueeze(logit, 1) # (batch_size, 1)
        logits = logits.repeat(1, self.num_experts) # (batch_size, num_experts)
        #print(logits.device)

        if mode == 'prop':
            expert_outputs = self.sigmoid(logits * self.a_prop + self.b_prop) # (batch_size, num_experts)
        else:
            expert_outputs = self.sigmoid(logits * self.a_imp + self.b_imp)
        #print(expert_outputs.device)
        
        expert_outputs = expert_outputs * selection_dist # (batch_size, num_experts)
        expert_outputs = expert_outputs.sum(1) # (batch_size, )
        #print(expert_outputs.device)
        
        # [0, 1]
        expert_outputs = expert_outputs - torch.lt(expert_outputs, 0) * expert_outputs #+ (expert_outputs < 0) * torch.ones_like(expert_outputs) * eps 
        expert_outputs = expert_outputs - torch.gt(expert_outputs, 1) * (expert_outputs - torch.ones_like(expert_outputs).cuda(self.gpu))
        
        return expert_outputs
        
    def _compute_IPS(self, x, num_epoch=200, lr=0.05, lamb=0, tol=1e-4, verbose=False):
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs) ## 전체 pair 개수 |D|
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
        
        for epoch in range(num_epoch):
            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                #print(x_all_idx.shape)
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda(self.gpu)
                #print(sub_obs.shape)

                prop_loss = nn.MSELoss()(prop, sub_obs)
                #prop_loss = nn.BCELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1 and verbose:
                print("[MF-DRJL-PS] Reach preset epochs, it seems does not converge.") 

    def _calibrate_IPS_G(self, x_val, x_test, num_epoch=100, lr=0.01, lamb=0, end_T=1e-3, verbose=False, G=10):
        x_all = generate_total_sample(self.num_users, self.num_items) # all (u,i) pairs = D
        obs = sps.csr_matrix((np.ones(x_val.shape[0]), (x_val[:, 0], x_val[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)

        t0 = time.time()
        self.propensity_model.eval()
        
        ## data prep
        ul_idxs = np.arange(x_all.shape[0]) # idxs
        np.random.shuffle(ul_idxs)
        neg_idxs = ul_idxs[:len(x_val) * G]
        
        ui_idxs = np.concatenate((x_all[neg_idxs], x_val), axis=0) # (u,i) pairs
        sub_obs = torch.FloatTensor(np.concatenate((obs[neg_idxs], np.ones(len(x_val))), axis=0)).cuda(self.gpu) ## y (label)
        
        ## fit CE - Adam
        optimizer = torch.optim.Adam([self.a_prop, self.b_prop, self.prop_selection_net[0].weight], lr=lr, weight_decay=lamb)
        nll_criterion = nn.CrossEntropyLoss().cuda(self.gpu)

        total_batch = len(ui_idxs) // self.batch_size_prop
        current_T = torch.tensor(1.).cuda(self.gpu)
        for epoch in range(num_epoch):
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                batch_idx = ui_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                batch_y = sub_obs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]      
                
                prop = self.calibration_experts(batch_idx, current_T, mode='prop')
                prop_loss = nll_criterion(prop, batch_y)

                optimizer.zero_grad()
                prop_loss.backward()
                optimizer.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            current_T = torch.tensor(1. * ((end_T / 1.) ** (epoch / num_epoch))).cuda(self.gpu)
            
            if verbose:
                if epoch % 10 == 0:    
                    print("epoch:", epoch, "loss:", epoch_loss)
                    #print("a_prop:", self.a_prop, "b_prop:", self.b_prop)
        
        
        # ## fit CE - LBFGS
        # optimizer = torch.optim.LBFGS([self.a_prop, self.b_prop, self.prop_selection_net[0].weight], lr=lr, max_iter=50)    
        # def eval():
        #     optimizer.zero_grad()
        #     prop = self.calibration_experts(batch_idx, current_T, mode='prop')
        #     loss = nll_criterion(prop, batch_y)
        #     loss.backward()
        #     return loss
        # optimizer.step(eval)
        
        if verbose:
            print("calibraton done in", time.time() - t0)
            
            ## test ECE
            obs = sps.csr_matrix((np.ones(x_test.shape[0]), (x_test[:, 0], x_test[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)

            ul_idxs = np.arange(x_all.shape[0]) # idxs
            np.random.shuffle(ul_idxs)
            neg_idxs = ul_idxs[:len(x_test) * G]
            
            ui_idxs = np.concatenate((x_all[neg_idxs], x_test), axis=0) # (u,i) pairs            
            sub_obs = np.concatenate((obs[neg_idxs], np.ones(len(x_test))), axis=0)

            self.a_prop.requires_grad = False
            self.b_prop.requires_grad = False
            self.prop_selection_net.requires_grad = False
            scores_uncal = self.propensity_model.forward(ui_idxs).detach().cpu()
            scores_cal = self.calibration_experts(ui_idxs, current_T, mode='prop').detach().cpu()
            
            print(scores_uncal.mean(), scores_uncal.std())
            print(scores_uncal)
            print(scores_cal.mean(), scores_cal.std())
            print(scores_cal)
            
            ECE_uncal = ECELoss(scores_uncal, torch.LongTensor(sub_obs))
            ECE_cal = ECELoss(scores_cal, torch.LongTensor(sub_obs))
            print("test ECE:", ECE_uncal, ECE_cal)

    def fit(self, x, y, x_val, y_val, stop = 5, num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1, tol=1e-4, G=1, end_T=1e-3, lr_imp=0.05, lamb_imp=0, lr_impcal=0.05, lamb_impcal=0, iter_impcal=10, verbose=False, cal=True): 
        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(self.imputation_model.parameters(), lr=lr_imp, weight_decay=lamb_imp)
        optimizer_impcal = torch.optim.Adam([self.a_imp, self.b_imp, self.imp_selection_net[0].weight], lr=lr_impcal, weight_decay=lamb_impcal)
        self.propensity_model.eval()
        
        x_all = generate_total_sample(self.num_users, self.num_items) ## D
        num_sample = len(x) # O
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        current_T = torch.tensor(1.).cuda(self.gpu)
        for epoch in range(num_epoch): 
            # O
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # D
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            for idx in range(total_batch):
                ## prediction model update
                self.prediction_model.train()
                self.imputation_model.eval()
                # O part (if o_ui=1)
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda(self.gpu)
                
                inv_prop = 1/torch.clip(self.calibration_experts(sub_x, 1e-3, mode='prop').detach(), gamma, 1)
                
                pred = self.prediction_model.forward(sub_x)

                #imputation_y = self.imputation_model.predict(sub_x).cuda(self.gpu) 
                imputation_y = self.calibration_experts(sub_x, current_T, mode='imp')
                              
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # e/p
                imputation_loss = F.binary_cross_entropy(pred, torch.clip(imputation_y,0,1), reduction="sum") # e^
                ips_loss = (xent_loss - imputation_loss) # batch size, e/p - e^ (current) <<<<===>>>> e/p - e^/p + e^ (paper)

                # D part (if o_ui=0)
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]] # negative ratio=G
                
                pred_u = self.prediction_model.forward(x_sampled) 
                #imputation_y1 = self.imputation_model.predict(x_sampled).cuda(self.gpu) 
                imputation_y1 = self.calibration_experts(x_sampled, current_T, mode='imp')
                direct_loss = F.binary_cross_entropy(pred_u, torch.clip(imputation_y1,0,1), reduction="sum")
    
                # total loss
                loss = ips_loss/self.batch_size + direct_loss/self.batch_size #/G

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                                           
                epoch_loss += xent_loss.detach().cpu().numpy()              
                
                ## imputation model update (O)
                self.prediction_model.eval()
                self.imputation_model.train()
                pred = self.prediction_model.predict(sub_x).cuda(self.gpu) ## prediction: y_hat
                imputation_y = self.imputation_model.forward(sub_x) ## pseudo label: y_tilde
                #imputation_y = self.calibration_experts(sub_x, current_T, mode='imp') ## 안하는게 좋다??
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none") ## actual loss: e
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none") ## imputed loss: e_hat
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum() ## error deviation: (e - e_hat)^2 / p  -> loss function for imputation model

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()

            ## imputation model calibration (O) + Propensity (하는게 성능 더 좋긴해 in coat)
            if cal:
                self.imputation_model.eval()
                inv_prop = 1/torch.clip(self.calibration_experts(x_val, 1e-3, mode='prop').detach(), gamma, 1)
                
                for i in range(iter_impcal):
                    prop = self.calibration_experts(x_val, current_T, mode='imp')
                    prop_loss = F.binary_cross_entropy(prop, torch.FloatTensor(y_val).cuda(self.gpu), weight=inv_prop, reduction="sum")

                    optimizer_impcal.zero_grad()
                    prop_loss.backward()
                    optimizer_impcal.step()
                    
                current_T = torch.tensor(1. * ((end_T / 1.) ** (epoch / num_epoch))).cuda(self.gpu)

            ## early stop
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1 and verbose:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")
                
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()