import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

def toymd(time):
    return datetime.utcfromtimestamp(time)#.strftime('%Y-%m-%d')

class PERIS(nn.Module):
    def __init__(self, opt):
        super(PERIS, self).__init__()       
        self.numuser = opt.numuser
        self.numitem = opt.numitem

        # ----------- Hyperparameters -----------        
        self.lamb = opt.lamb        
        self.mu = opt.mu                           
        
        self.tau = opt.tau          
        self.bin_ratio = opt.bin_ratio
        self.neg_weight = opt.neg_weight
        self.aggtype = opt.aggtype                                

        self.margin = opt.margin # HP for perference learning (CML)
        self.ebd_size = opt.K 
        # -------- End of Hyperparameters --------

        NUM_PROTOTYPE=3

        self.ebd_user = nn.Embedding(self.numuser+1, self.ebd_size).cuda()
        self.ebd_item = nn.Embedding(self.numitem+1, self.ebd_size).cuda()
        self.ebd_prototype = nn.Embedding(NUM_PROTOTYPE, self.ebd_size).cuda() 

        nn.init.xavier_normal_(self.ebd_user.weight)
        nn.init.xavier_normal_(self.ebd_item.weight)
        nn.init.xavier_normal_(self.ebd_prototype.weight)        

        self.user_idx = torch.LongTensor([0]).cuda()
        self.neighbor_idx = torch.LongTensor([1]).cuda()
        self.consumption_idx = torch.LongTensor([0]).cuda()        
        
        self.clip_max = torch.FloatTensor([1.0]).cuda()     
        
      

        self.bce = nn.BCELoss(reduction='none')

        # For end-to-end learning with covering the test case
        trntimes = np.load(opt.dataset_path + '/trn')[:,-1].astype(int) # ascending sorted
        vld = [l for l in csv.reader(open('/'.join(opt.dataset_path.split('/')[:-1])+'/split/vld.csv'))]
        vldtimes = np.array(vld)[:,-1].astype(int)                       
        
        trnfront_time = trntimes.max() - 60 * 60 * 24 * 7 * opt.period 
        trnfront_idx = np.where(trntimes < trnfront_time)[0][-1]        
        self.label_period_start_time = trntimes[trnfront_idx] # for training
        self.trn_end_time = max(trntimes) # for validation
        self.vld_end_time = max(vldtimes) # for test        

        self.hidden_dim = self.ebd_size        
        
        self.lstm = nn.LSTM(
            input_size=1,            
            hidden_size=self.hidden_dim, 
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.proj = nn.Linear(self.hidden_dim, 1, bias=False)        
        nn.init.xavier_normal_(self.proj.weight)

        self.proj_item = nn.Linear(self.hidden_dim, 1, bias=False)        
        nn.init.xavier_normal_(self.proj_item.weight) 
            
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)            
        

        
    def init_embeddings(self):        
        # Unit-sphere restriction
        user_weight = self.ebd_user.weight.data
        user_weight.div_(torch.max(torch.norm(user_weight, 2, 1, True),
                                   self.clip_max).expand_as(user_weight))

        item_weight = self.ebd_item.weight.data
        item_weight.div_(torch.max(torch.norm(item_weight, 2, 1, True),
                                   self.clip_max).expand_as(item_weight))

        pro_weight = self.ebd_prototype.weight.data
        pro_weight.div_(torch.max(torch.norm(pro_weight, 2, 1, True),
                                   self.clip_max).expand_as(pro_weight))  
    
    def aggregate_labels(self, scores, mask4label):
        if self.aggtype == 'sum':
            virtlabels = scores.sum(-1)
        elif self.aggtype == 'mean':
            denorm = mask4label.sum(-1)
            denorm[denorm==0] = 1
            virtlabels = scores.sum(-1) / denorm[:, None]
        elif self.aggtype == 'max':
            virtlabels = scores.max(-1)[0]
            
        virtlabels[virtlabels>1] = 1
            
        return virtlabels 
    
    def compute_similarity(self, A, B): # From Bc x |A| x d and Bc x |B| x d to Bc x |A| x |B|
        if len(A.shape) == 3: # For the embeddings for item sequneces
            hadamard = (A[:, None, :, :] * B[:, :, None, :]).sum(-1) 
            cos = hadamard / (A.norm(dim=-1)[:, None, :] * B.norm(dim=-1)[:, :, None])            
            
        elif len(A.shape) == 2: # for the embeddings for useres (not sequence)
            hadamard = (A[None, :, :] * B[:, None, :]).sum(-1)
            cos = hadamard / (A.norm(dim=-1)[None, :] * B.norm(dim=-1)[:, None])              
        else: # Exception            
            print('ERROR while computing the simiarity!')            
            
        sim = (cos+1)/2 + self.tau   
        
        sim[sim>1] = 1       
        
        return sim
    
    def encode_timebin(self, timebins): # encoding time bins using RNN    
        
        if len(timebins.shape) == 3:                 
            flat_timebins = timebins.reshape(-1, timebins.shape[-1])        
            
            input_lengths = flat_timebins.sum(-1).bool().float() * flat_timebins.size(1)            
            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            sorted_flat_timebins = flat_timebins[sorted_idx]

            packed_input = nn.utils.rnn.pack_padded_sequence(sorted_flat_timebins.unsqueeze(-1), input_lengths.clamp(min=1).tolist(), batch_first=True)

            packed_output, hidden = self.lstm(packed_input)

            output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            feature_timebins = output[:, -1, :]            

            feature_timebins = feature_timebins.reshape(timebins.shape[0], timebins.shape[1], -1)        

        else: 
            feature_timebins = self.lstm(timebins.float().unsqueeze(-1))[0][:,-1,:]
 
        
        return feature_timebins
        

    # Intrinsic supplementaion scheme
    def virtual_labeling(self, users, emb_users, items, emb_items, history, pack_times, times, time_bins, emb_eval_items, mode): # mode: trn/vld/tst                 
        # NOTE: The preprocessing code for time bining is in data_loader
        
        if emb_eval_items != None: # Similarity-based feature generation for evaluation                    
            sim = self.compute_similarity(emb_items, emb_eval_items)
            
            mask4feature = (items != 0)[:,None,:]        
            features = (sim * mask4feature) @ time_bins      
            
            return None, features
        
        # Divide items based on the start time of the label period
        mask = (pack_times != 0)
        items4predict = ((pack_times < self.label_period_start_time) * mask).float()
        items4label = ((pack_times >= self.label_period_start_time) * mask).float()
        org_items4label = (times >= self.label_period_start_time) #* mask
        
        # 3. Virtual labeling for each item
        # Matrix multiplication using [items4predict + items4label + paddings] in a one shot
                
        # 3.1 Building a similarity matrix                
        emb_history = self.ebd_item(history)

        # NOTE: tensor-wise similarity computation         
        hadamard = (emb_items[:, None, :] * emb_history).sum(-1)
        cos = hadamard / (emb_items.norm(dim=-1)[:, None] * emb_history.norm(dim=-1))                      
        sim = (cos+1)/2 + self.tau     

        sim[sim>1] = 1 

        # 3.2 Generating features with considering the simiarity among items
        # NOTE: time bins of label items should be excluded to avoid the trivial prediction. 
        mask_featuring = mask * items4predict
        features = (sim * mask_featuring)[:,None,:] @ time_bins  
        features = features.squeeze()  
        
        virtlabels = org_items4label.float()
        return virtlabels, features


    def neighbor_feature_label(self, users, neighbors, neighbors_time, neighbors_freqbin, mode): # TODO: add mode to hance evluation
        # 1. user - candidate user similarity
        # 2. aggregate frequency bins based on the simialrity with the feature mask
        # 3. compute the labels
        emb_user = self.ebd_user(users)
        emb_neighbors = self.ebd_user(neighbors)

        if mode == 'trn':
            feature_time = self.label_period_start_time
        else:
            emb_user = torch.repeat_interleave(emb_user, torch.LongTensor([101]*len(users)).cuda(), dim=0)            

            if mode == 'vld':
                feature_time = self.trn_end_time
            elif mode == 'tst':
                feature_time = self.vld_end_time

        # Similarity computaion
        hadamard = (emb_user[:, None, :] * emb_neighbors).sum(-1)
        cos = hadamard / (emb_user.norm(dim=-1)[:, None] * emb_neighbors.norm(dim=-1))                      
        sim = (cos+1)/2 + self.tau
        sim[sim>1] = 1

        mask = (neighbors!=0) 
        
        # Divide items based on the start time of the label period                
        items4predict = ((neighbors_time < feature_time) * mask).float()
        labels = ((neighbors_time >= feature_time) * sim * mask).float()
        labels = (labels.sum(-1) > 1).float()      
 
        mask_featuring = mask * items4predict 

        features = (sim * mask_featuring)[:,None,:] @ neighbors_freqbin.float()  

        features = features.squeeze()   

        return labels, features
    
    def shorten_timebins(self, timebins):
        numtimebins = timebins.shape[-1]
        half_numtb = int(numtimebins * self.bin_ratio)
        if len(timebins.shape) == 3:
            timebins = timebins[:,:,-half_numtb:]
        elif len(timebins.shape) == 2:
            timebins = timebins[:,-half_numtb:]
        else:
            print('Error while reducing timebins!')
            exit()
        return timebins        
                
    def compute_loss(self, batch_data):
        self.init_embeddings()        

        users, items, times, user_timebins, history, pack_times, all_neg_items, item_timebins, item_labels, neighbors, neighbors_time, neighbors_freqbin = batch_data

        emb_user = self.ebd_user(users)
        emb_item = self.ebd_item(items)                
        all_emb_neg_item = self.ebd_item(all_neg_items) 

        prot = self.ebd_prototype(self.user_idx)
        prot_neighbor = self.ebd_prototype(self.neighbor_idx)
        consumption = self.ebd_prototype(self.consumption_idx)

        virtlabels, user_timebins = self.virtual_labeling(users, emb_user, items, emb_item, history, pack_times, times, user_timebins, None, mode='trn')

        neighbor_labels, neighbor_timebins = self.neighbor_feature_label(users, neighbors, neighbors_time, neighbors_freqbin, mode='trn')

        # The number of timebins is controlled by bin_ratio hyper-parameter
        neighbor_timebins = self.shorten_timebins(neighbor_timebins)
        user_timebins = self.shorten_timebins(user_timebins)

        feature_timebins_user = self.encode_timebin(user_timebins)               
        
        feature_timebins_neighbor = self.encode_timebin(neighbor_timebins)

        up_feature = emb_user + emb_item       
        user_pattern = feature_timebins_user + up_feature
        neighbor_pattern = feature_timebins_neighbor 

        u_dist = (prot - user_pattern).pow(2).sum(-1).sqrt()            
        ui_prob = torch.max(1 - u_dist, torch.FloatTensor([0]).cuda())     
       
        neighbor_dist = (prot_neighbor - neighbor_pattern).pow(2).sum(-1).sqrt()    
        n_prob = torch.max(1 - neighbor_dist, torch.FloatTensor([0]).cuda()) 
    
        loss_UIS = torch.pow(ui_prob - virtlabels, 2) 
        loss_NIS = torch.pow(n_prob - neighbor_labels, 2) 

        # Apply a weight for negative class (w \in [0,1])
        class_weight = torch.ones(virtlabels.shape).cuda() # Positives are set to 1
        class_weight[virtlabels == 0] = self.neg_weight     

        class_weight_neg = torch.ones(neighbor_labels.shape).cuda() # Positives are set to 1
        class_weight_neg[neighbor_labels == 0] = self.neg_weight     

        loss_UIS = loss_UIS * class_weight
        loss_NIS = loss_NIS * class_weight_neg         

        loss_NIS = loss_NIS.mean()

        loss_UIS = loss_UIS.mean()           


        
        loss_pref = self.get_pref_loss(users, items, all_neg_items) # It include the masking operation
    
        loss = self.lamb * loss_UIS + self.mu *loss_NIS + loss_pref

        loss = self.lamb * (self.mu * loss_UIS + (1-self.mu) * loss_NIS) + (1-self.lamb) * loss_pref 

        loss_IS4debug = loss_UIS        

        return loss, loss_IS4debug # loss_IS is a redudant loss so it's okay
    
    def compute_warmup_loss(self, batch_data):        
        users, items, _, _, _, _, all_neg_items, _, _, _, _, _ = batch_data
        return self.get_pref_loss(users, items, all_neg_items) 

    def get_pref_loss(self, users, items, all_neg_items):
        self.init_embeddings()
        
        emb_user = self.ebd_user(users)
        emb_item = self.ebd_item(items)
        all_emb_neg_item = self.ebd_item(all_neg_items)     

        consumption = self.ebd_prototype(self.consumption_idx) 

        mask = (items != 0)

        # Prototype computation
        up_feature = emb_user + emb_item
        un_feature = emb_user[None,:,:] + all_emb_neg_item
        c_dist_p = (consumption - up_feature).pow(2).sum(-1).sqrt()
        all_c_dist_n = (consumption[None, :, :] - un_feature).pow(2).sum(-1).sqrt() 
        loss_PL = torch.max(c_dist_p[None, :] - all_c_dist_n + self.margin, torch.FloatTensor([0]).cuda()) 
        loss_PL = loss_PL.mean(dim=0)
       
        loss_each, loss_agg = (loss_PL * mask), (loss_PL * mask).sum()/mask.sum()

        return loss_agg
    
    def predict_for_multiple_item(self, userinfo, cand_items, mode, item_timebins, neighbors, neighbors_time, neighbors_freqbin):        
        users, items, times, user_timebins = [ui.cuda() for ui in userinfo]
        cand_items = cand_items.cuda()      

        emb_user = self.ebd_user(users)
        emb_item = self.ebd_item(items)
        emb_cand_items = self.ebd_item(cand_items)   

        prot = self.ebd_prototype(self.user_idx)
        prot_neighbor = self.ebd_prototype(self.neighbor_idx)      
        consumption = self.ebd_prototype(self.consumption_idx)       
                
        _, user_timebins = self.virtual_labeling(users, emb_user, items, emb_item, None, None, times, user_timebins, emb_cand_items, mode)

        neighbors, neighbors_time, neighbors_freqbin = neighbors.cuda(), neighbors_time.cuda(), neighbors_freqbin.cuda()
        _, neighbor_timebins = self.neighbor_feature_label(users, neighbors, neighbors_time, neighbors_freqbin, mode)

        # The number of timebins is controlled by bin_ratio hyper-parameter
        user_timebins = self.shorten_timebins(user_timebins)
        neighbor_timebins = self.shorten_timebins(neighbor_timebins)

        feature_timebins_user = self.encode_timebin(user_timebins.cuda())
        feature_timebins_neighbor = self.encode_timebin(neighbor_timebins.cuda())
    
        ui_feature = emb_user[:,None,:] + emb_cand_items 

        user_pattern = feature_timebins_user + ui_feature
        neighbor_pattern = feature_timebins_neighbor

        c_dist = (consumption[None, :,:] - ui_feature).pow(2).sum(-1).sqrt()

        ui_dist = (prot[:,None,:] - user_pattern).pow(2).sum(-1).sqrt()            
       
        n_dist = (prot_neighbor - neighbor_pattern).pow(2).sum(-1).sqrt().reshape(c_dist.shape)   

        c_score, ui_score, n_score = (1-c_dist), (1-ui_dist), (1-n_dist)

        score = self.lamb * (self.mu * ui_score + (1-self.mu) * n_score) + (1-self.lamb) * c_score 
        
        return score
    
    def get_pis(self, userinfo, cand_items, mode, item_timebins, neighbors, neighbors_time, neighbors_freqbin):        
        users, items, times, user_timebins = [ui.cuda() for ui in userinfo]
        cand_items = cand_items.cuda()      

        emb_user = self.ebd_user(users)
        emb_item = self.ebd_item(items)
        emb_cand_items = self.ebd_item(cand_items)   

        prot = self.ebd_prototype(self.user_idx)
        prot_neighbor = self.ebd_prototype(self.neighbor_idx)      
        consumption = self.ebd_prototype(self.consumption_idx)       
                
        _, user_timebins = self.virtual_labeling(users, emb_user, items, emb_item, None, None, times, user_timebins, emb_cand_items, mode)

        neighbors, neighbors_time, neighbors_freqbin = neighbors.cuda(), neighbors_time.cuda(), neighbors_freqbin.cuda()
        _, neighbor_timebins = self.neighbor_feature_label(users, neighbors, neighbors_time, neighbors_freqbin, mode)

        # The number of timebins is controlled by bin_ratio hyper-parameter
        user_timebins = self.shorten_timebins(user_timebins)
        neighbor_timebins = self.shorten_timebins(neighbor_timebins)

        feature_timebins_user = self.encode_timebin(user_timebins.cuda())
        feature_timebins_neighbor = self.encode_timebin(neighbor_timebins.cuda())
    
        ui_feature = emb_user[:,None,:] + emb_cand_items 

        user_pattern = feature_timebins_user + ui_feature
        neighbor_pattern = feature_timebins_neighbor #+ ui_feature.reshape(-1, ui_feature.shape[-1])

        c_dist = (consumption[None, :,:] - ui_feature).pow(2).sum(-1).sqrt()

        ui_dist = (prot[:,None,:] - user_pattern).pow(2).sum(-1).sqrt()            
       
        n_dist = (prot_neighbor - neighbor_pattern).pow(2).sum(-1).sqrt().reshape(c_dist.shape)   

        ui_score, n_score = (1-ui_dist), (1-n_dist)

        pis = (self.mu * ui_score + (1-self.mu) * n_score)

        return pis