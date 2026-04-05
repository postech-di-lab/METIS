import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace

from utils.data_loader import *
from utils.evaluation_all import *
from utils.util import *
from utils.early_stop import *

from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim = args.dim
        self.client_param = {}
        
        self.item_emb = nn.Embedding(self.args.num_item, self.dim)
        self.affine_output = nn.Linear(self.dim, 1)
        
    
    def forward(self, item_idx):
        item_emb = self.item_emb(item_idx)
        logits = self.affine_output(item_emb)
        return logits
    
    def get_score_mat_only_user_grad_true(self):
        item_emb = self.item_emb.weight.data # no gradient
        logits = self.affine_output(item_emb)
        return logits    
    
    def grad_on(self):
        self.item_emb.weight.requires_grad = True

        for param in self.affine_output.parameters():
            param.requires_grad = True
    
    def get_score_mat(self):
        if self.args.specific_gpu == 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{self.args.gpu}")
        
        server_affine_weight = copy.deepcopy(self.affine_output.state_dict())         
        score = torch.zeros(self.num_user, self.num_item)
        
        for user in self.client_param.keys():
            self.affine_output.load_state_dict({k.replace('affine_output.', ''):v for k, v in self.client_param[user].items() if 'affine_output' in k})

            prediction = self.affine_output(self.item_emb.weight.data.view(self.num_item, self.args.dim))
            score[user] = prediction.view(-1)
        
        self.affine_output.load_state_dict(server_affine_weight)

        return score.clone().detach()
    
    
    def get_score_mat_only_user_grad_true(self):
        item_emb_tensor = self.item_emb.weight.data
        
        pred_mat = self.affine_output(item_emb_tensor)
        
        return pred_mat


class PFedRec_Engine(nn.Module):
    def __init__(self, args, device):
        super(PFedRec_Engine, self).__init__()
        
        self.args = args
        self.device = device
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim = args.dim
        
        
        self.model = MLP(self.args)
        
        self.client_model_params = {}
        self.best_client_model_params = {}
        
        # emb_reg
        self.old_item_emb = nn.Embedding(self.num_item, self.dim)
        self.old_user_size = args.num_user
        
        # Metric Matrix
        self.A_N20 = torch.zeros((args.num_task, args.num_task))
        self.A_R20 = torch.zeros((args.num_task, args.num_task))
        
        self.train_mat_dict = {}
        self.train_mat_tensor = {}
        self.valid_mat_tensor = {}
        self.test_mat_tensor = {}
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        
    def add_user_item(self, new_num_user, new_num_item):
        if self.num_item != new_num_user:
            self.num_user = new_num_user
            self.model.num_user = new_num_user
            
        if self.num_item != new_num_item:
            new_item_emb = nn.Embedding(new_num_item - self.num_item, self.dim)
            new_weights = torch.cat((self.model.item_emb.weight, new_item_emb.weight), dim=0).clone().detach()
            item_emb = nn.Embedding.from_pretrained(new_weights)
            
            self.num_item = new_num_item
            self.model.item_emb = nn.Embedding.from_pretrained(item_emb.weight.clone().detach())
            self.model.num_item = new_num_item


    
    def validate(self, train_mat_tensor, test_mat_tensor):
        return evaluate(self.model, train_mat_tensor, test_mat_tensor, self.device)
    
        
    def test(self, task):
        # 1. a_{i,i}
        masking = self.train_mat_tensor[f"TASK_{task}"] + self.valid_mat_tensor[f"TASK_{task}"]
        RESULT = self.validate(masking, self.test_mat_tensor[f"TASK_{task}"])
        N_20, R_20 = RESULT["N@20"], RESULT["R@20"]
        self.A_N20[task][task] = N_20
        self.A_R20[task][task] = R_20
    
        
        # 2. a_{task, i}
        for i in range(task): # i: 0 ~ task-1
            masking = self.train_mat_tensor[f"TASK_{task}"] + self.valid_mat_tensor[f"TASK_{task}"]
            RESULT = self.validate(masking, self.test_mat_tensor[f"TASK_{i}"])
            N_20, R_20 = RESULT["N@20"], RESULT["R@20"]
            self.A_N20[task][i] = N_20
            self.A_R20[task][i] = R_20
        
        return get_LA_RA(task, self.A_N20, self.A_R20)
        
    
    def task_specific_data_processing(self, task, input_total_data):
        total_train_dataset, total_valid_dataset, total_test_dataset = input_total_data
        
        for i in range(task + 1):
            self.train_mat_dict[f"TASK_{i}"], self.train_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_train_dataset[f"TASK_{i}"], True, self.num_user, self.num_item)
            self.valid_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_valid_dataset[f"TASK_{i}"], False, self.num_user, self.num_item)
            self.test_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_test_dataset[f"TASK_{i}"], False, self.num_user, self.num_item)
    

    def run(self, task, input_total_data, is_base):
        self.task_specific_data_processing(task, input_total_data)
        
        
        block_info = f"Inc #{task} Block!" if task != 0 else "Base Block!"
        
        print('\n ==========' + block_info + '==========')
        
        best_valid = -torch.inf
        best_param = self.model.state_dict()
        self.best_client_model_params = copy.deepcopy(self.client_model_params)
        best_epoch = -1
        
        early = EarlyStopping(patience = self.args.patience, larger_is_better=True)
                
        for round in range(self.args.num_round):
            self.fed_train_a_round(task, is_base, round)
            
            self.model.client_param = copy.deepcopy(self.client_model_params)
            RESULT = self.validate(self.train_mat_tensor[f"TASK_{task}"], self.valid_mat_tensor[f"TASK_{task}"])
            self.model.client_param={}
            N_20, R_20 = RESULT['N@20'], RESULT['R@20']
            epoch_val = N_20
            
            if self.args.show_valid:
                print(f"Epoch: {round + 1} N@20: {N_20:.5f} R@20: {R_20:.5f}")
            
            if epoch_val > best_valid:
                best_valid = epoch_val
                best_param = self.model.state_dict()  
                self.best_client_model_params = copy.deepcopy(self.client_model_params)
                best_epoch = round + 1

            early(epoch_val)
            
            if early.stop_flag:
                break
            
        # test phase
        self.model.load_state_dict(best_param)
        self.model.client_param = copy.deepcopy(self.best_client_model_params)
        RESULT = self.test(task)
        
        self.client_model_params = copy.deepcopy(self.best_client_model_params)
        
        print(f"Best Epoch: {best_epoch}")
        print(f"TEST N@20: {self.A_N20[task][task]:.5f} R@20: {self.A_R20[task][task]:.5f}")
        

        if self.args.save_result ==1 and not is_base:
            save_result_as_csv(self.args, self.args.baseline, RESULT, task)

        # topn_list ============================================================================================
        self.model.to(self.device)
        self.model.train()
        self.target_mat = self.model.get_score_mat().data.to('cpu')
        self.model.to('cpu')
        self.model.client_param = {}
        # ==========================================================
        
        if is_base:
            __, self.topn_list = torch.topk(self.target_mat, self.args.topN, dim=1)
        
        else: 
            diff_num_user = self.num_user - len(self.topn_list)
            tmp_list = torch.zeros((diff_num_user, self.args.topN), dtype=torch.long)
            self.topn_list = torch.concat((self.topn_list, tmp_list), dim=0)
            
            trained_user_in_this_block = torch.tensor(list(self.train_mat_dict[f"TASK_{task}"].keys()))
            ___, block_topn_list = torch.topk(self.target_mat[trained_user_in_this_block], self.args.topN, dim=1)
            self.topn_list[trained_user_in_this_block] = block_topn_list        
            
        # assign old embedding ============================================================================================
        self.old_item_emb = nn.Embedding.from_pretrained(self.model.item_emb.weight.clone().detach())
        self.old_user_size = self.num_user
        
            
    def fed_train_a_round(self, task, is_base, round):
        
        train_mat_dict = self.train_mat_dict[f"TASK_{task}"]
        num_participants = int( len(list(train_mat_dict.keys())) * self.args.clients_sample_ratio)
        participants = random.sample(list(train_mat_dict.keys()), num_participants)
        round_user_params = {}
            
        
        server_param = copy.deepcopy(self.model.state_dict()['item_emb.weight'].data.cpu())
            
        # train
        for user in tqdm(participants, disable=not self.args.tqdm):    
            client_model = copy.deepcopy(self.model)
            if round != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                
                if user in self.client_model_params.keys(): # mlp
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key])
                user_param_dict['item_emb.weight'] = copy.deepcopy(server_param)
                client_model.load_state_dict(user_param_dict)
                
            client_model.to(self.device)


            if self.args.optimizer == "SGD":
                optimizer = torch.optim.SGD(client_model.affine_output.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
                optimizer_i = torch.optim.SGD(client_model.item_emb.parameters(), lr = self.args.lr * self.num_item * self.args.lr_eta, weight_decay=self.args.weight_decay)
            elif self.args.optimizer == "Adam":
                optimizer = torch.optim.Adam(client_model.affine_output.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
                optimizer_i = torch.optim.Adam(client_model.item_emb.parameters(), lr = self.args.lr * self.num_item * self.args.lr_eta, weight_decay=self.args.weight_decay)
                
            optimizers = [optimizer, optimizer_i]     
    
            user_dataloader = DataLoader(dataset = Pos_Neg_Sampler(self.args, train_mat_dict[user]), batch_size=self.args.batch_size, shuffle = True)
            client_model.train()
            
            for epoch in range(self.args.local_epoch):
                for batch in user_dataloader:
                    client_model = self.fed_train_single_user(client_model, batch, optimizers, round, user, is_base)                    
            
            client_model.to('cpu')
            client_param = client_model.state_dict()
            
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.to('cpu')
                    
            round_user_params[user] = copy.deepcopy(client_model.item_emb.weight.data)
            if self.args.dp > 0:
                round_user_params[user] += Laplace(0, self.args.dp).expand(round_user_params[user].shape).sample()
            
        
        self.aggregate_clients_params(round_user_params, is_base, round)
                         
        
    def fed_train_single_user(self, client_model, batch_data, optimizers, round, user, is_base):
        
        # 1-step
        client_model.grad_on()
        
        items, target = batch_data[0].to(self.device), batch_data[1].float().to(self.device)
        optimizer, optimizer_i = optimizers
        optimizer.zero_grad()
        prediction = client_model(items)
        loss = self.loss_fn(prediction.view(-1), target)
        
        # user part
        if not is_base and self.args.client_cl and user < self.old_user_size:
            target_mat = self.target_mat[user].to(self.device)
            target_mat.requires_grad = False
            
            item_idx = torch.tensor(torch.arange(self.num_item), device = self.device)
            pred_mat = client_model(item_idx).reshape(-1)
            
            last_topn_list = self.topn_list[user]
            kd_loss = get_rank_discrepancy_kd_loss(self.args, target_mat, pred_mat, last_topn_list, self.device)
            
            loss += kd_loss * self.args.reg_client_cl
        
        loss.backward()
        optimizer.step()
        
        
        # item part
        optimizer_i.zero_grad()
        prediction = client_model(items)
        loss_i = self.loss_fn(prediction.view(-1), target)
        
        if not is_base and self.args.client_cl and user < self.old_user_size:
            target_mat = self.target_mat[user].to(self.device)
            target_mat.requires_grad = False
            
            item_idx = torch.tensor(torch.arange(self.num_item), device = self.device)
            pred_mat = client_model(item_idx).reshape(-1)
            
            last_topn_list = self.topn_list[user]
            kd_loss = get_rank_discrepancy_kd_loss(self.args, target_mat, pred_mat, last_topn_list, self.device)
            
            loss_i += kd_loss * self.args.reg_client_cl
        
        loss_i.backward()
        optimizer_i.step()
        
    
        if not is_base and self.args.client_cl and user < self.old_user_size:
            optimizer_i.zero_grad()
            target_mat = self.target_mat[user].to(self.device)
            target_mat.requires_grad = False
            
            
            pred_mat = client_model.get_score_mat_only_user_grad_true().reshape(-1)
            
            last_topn_list = self.topn_list[user]
            kd_loss = get_rank_discrepancy_kd_loss(self.args, target_mat, pred_mat, last_topn_list, self.device)
            
            loss_i = kd_loss * self.args.reg_client_cl
        
            if loss_i > 0:
                loss_i.backward()
                optimizer_i.step()
        
        return client_model
        

    def aggregate_clients_params(self, round_user_params, is_base, round):
        # aggregate item embedding
        
        for idx, user in enumerate(round_user_params.keys()):
            
            user_params = round_user_params[user]
            
            if idx == 0:
                server_model_param = copy.deepcopy(user_params.data)
            else:
                server_model_param += user_params.data
        
        server_model_param /= len(round_user_params)
        
        if self.args.server_cl and (not is_base):
            old = self.old_item_emb.weight.data
            num_old_item = len(old)
            new_item_emb = server_model_param[num_old_item:]
            
            weight = self.args.beta * diff(old, server_model_param[:num_old_item])
            server_model_param = (1-weight) * server_model_param[:num_old_item] + weight * old # shape: old_item_emb * dim
            server_model_param = torch.concat((server_model_param, new_item_emb), dim=0)        # shape: num_item_emb * dim
        
        self.model.item_emb = nn.Embedding.from_pretrained(server_model_param)
        
        