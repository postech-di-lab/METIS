import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt

class BPR(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(BPR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.user_emb = nn.Embedding(self.num_user, emb_dim)
        self.item_emb = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

    # outputs logits
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        output = (pos_score, neg_score)
        
        return output

    def forward_pair(self, batch_user, batch_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        
        return pos_score     
    
    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]
        loss = -(pos_score - neg_score).sigmoid().log().sum()
        
        return loss

class UBPR(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, i_pop, gpu, eta):
        super(UBPR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.gpu = gpu
        self.eta = eta

        self.user_emb = nn.Embedding(self.num_user, emb_dim)
        self.item_emb = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
        
        self.i_propensity = torch.pow(i_pop / i_pop.max(), self.eta).cuda(gpu)

    def propensity(self, u, i):
        #return torch.pow(self.u_pop[u] / torch.max(self.u_pop), self.eta)
        propensities = self.i_propensity[i]

        return torch.max(propensities, torch.ones_like(propensities) * 0.1)
                
    # outputs logits
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        P_pos = self.propensity(batch_user, batch_pos_item)
        
        return pos_score, neg_score, P_pos

    def forward_pair(self, batch_user, batch_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        
        return pos_score     
    
    def get_loss(self, output):
        pos_score, neg_score, P = output[0], output[1], output[2]
        
        loss = -((pos_score - neg_score).sigmoid().log() / P).sum()
        
        return loss

class NeuMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, num_hidden_layer):
        super(NeuMF, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.user_emb_MF = nn.Embedding(self.num_user, emb_dim)
        self.item_emb_MF = nn.Embedding(self.num_item, emb_dim)

        self.user_emb_MLP = nn.Embedding(self.num_user, emb_dim)
        self.item_emb_MLP = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb_MF.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb_MF.weight, mean=0., std= 0.01)
		
        nn.init.normal_(self.user_emb_MLP.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb_MLP.weight, mean=0., std= 0.01)

		# Layer configuration
		##  MLP Layers
        MLP_layers = []
        layers_shape = [emb_dim * 2]
        for i in range(num_hidden_layer):
            layers_shape.append(layers_shape[-1] // 2)
            MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
            MLP_layers.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(* MLP_layers)
        print("MLP Layer Shape ::", layers_shape)
        
        ## Final Layer
        self.final_layer  = nn.Linear(layers_shape[-1]+emb_dim, 1)

        # Loss function
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        pos_score = self.forward_pair(batch_user, batch_pos_item)	 # bs x 1
        neg_score = self.forward_pair(batch_user, batch_neg_item)	 # bs x 1

        output = (pos_score, neg_score)

        return output

    def forward_pair(self, batch_user, batch_item):
        # MF
        u_mf = self.user_emb_MF(batch_user)			# batch_size x dim
        i_mf = self.item_emb_MF(batch_item)			# batch_size x dim
        
        mf_vector = (u_mf * i_mf)					# batch_size x dim

        # MLP
        u_mlp = self.user_emb_MLP(batch_user)		# batch_size x dim
        i_mlp = self.item_emb_MLP(batch_item)		# batch_size x dim

        mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vector = self.MLP_layers(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        output = self.final_layer(predict_vector) 

        return output

    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]

        pred = torch.cat([pos_score, neg_score], dim=0)
        gt = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        
        return self.BCE_loss(pred, gt)

class LightGCN(nn.Module):
	def __init__(self, user_count, item_count, dim, gpu, A, A_T):
		super(LightGCN, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).cuda(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).cuda(gpu)

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		self.A = A.to(gpu)	# user x item
		self.A_T = A_T.to(gpu)

		self.A.requires_grad = False
		self.A_T.requires_grad = False
		
		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
		
	def forward(self, user, pos_item, neg_item):
		u_0 = self.user_emb(self.user_list)	# num. user x dim
		i_0 = self.item_emb(self.item_list)

		i_1 = torch.spmm(self.A_T, u_0)		# 유저 평균 -> 아이템
		u_1 = torch.spmm(self.A, i_0)		# 아이템 평균 -> 유저
		
		i_2 = torch.spmm(self.A_T, u_1)
		u_2 = torch.spmm(self.A, i_1)		

		user_0 = torch.index_select(u_0, 0, user) # slice
		user_1 = torch.index_select(u_1, 0, user) 
		user_2 = torch.index_select(u_2, 0, user) 

		pos_0 = torch.index_select(i_0, 0, pos_item) 
		pos_1 = torch.index_select(i_1, 0, pos_item) 
		pos_2 = torch.index_select(i_2, 0, pos_item) 

		neg_0 = torch.index_select(i_0, 0, neg_item) 
		neg_1 = torch.index_select(i_1, 0, neg_item) 
		neg_2 = torch.index_select(i_2, 0, neg_item) 

		u = (user_0 + user_1 + user_2) / 3
		i = (pos_0 + pos_1 + pos_2) / 3
		j = (neg_0 + neg_1 + neg_2) / 3

		pos_score = (u * i).sum(dim=1, keepdim=True)
		neg_score = (u * j).sum(dim=1, keepdim=True)

		return (pos_score, neg_score)

	def get_embedding(self):

		u_0 = self.user_emb(self.user_list)	# num. user x dim
		i_0 = self.item_emb(self.item_list)

		i_1 = torch.spmm(self.A_T, u_0)		# 유저 평균 -> 아이템
		u_1 = torch.spmm(self.A, i_0)		# 아이템 평균 -> 유저
		
		i_2 = torch.spmm(self.A_T, u_1)
		u_2 = torch.spmm(self.A, i_1)	

		user = (u_0 + u_1 + u_2) / 3
		item = (i_0 + i_1 + i_2) / 3

		return user, item

	def get_loss(self, output):
		pos_score, neg_score = output[0], output[1]
		loss = -(pos_score - neg_score).sigmoid().log().sum()
		
		return loss

	def forward_pair(self, user, item):
		u_0 = self.user_emb(self.user_list)	# num. user x dim
		i_0 = self.item_emb(self.item_list)

		i_1 = torch.spmm(self.A_T, u_0)		# 유저 평균 -> 아이템
		u_1 = torch.spmm(self.A, i_0)		# 아이템 평균 -> 유저
		
		i_2 = torch.spmm(self.A_T, u_1)
		u_2 = torch.spmm(self.A, i_1)		

		user_0 = torch.index_select(u_0, 0, user) # slice
		user_1 = torch.index_select(u_1, 0, user) 
		user_2 = torch.index_select(u_2, 0, user) 

		pos_0 = torch.index_select(i_0, 0, item) 
		pos_1 = torch.index_select(i_1, 0, item) 
		pos_2 = torch.index_select(i_2, 0, item) 

		u = (user_0 + user_1 + user_2) / 3
		i = (pos_0 + pos_1 + pos_2) / 3

		pos_score = (u * i).sum(dim=1, keepdim=True)

		return pos_score

class CML(nn.Module):
	def __init__(self, user_count, item_count, dim, margin, gpu):
		super(CML, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)
			
		# User / Item Embedding
		self.user_emb = nn.Embedding(self.user_count, dim, max_norm=1.) #
		self.item_emb = nn.Embedding(self.item_count, dim, max_norm=1.) #

		nn.init.normal_(self.user_emb.weight, mean=0., std= 1 / (dim ** 0.5))
		nn.init.normal_(self.item_emb.weight, mean=0., std= 1 / (dim ** 0.5))

		self.margin = margin

	def forward(self, batch_user, batch_pos_item, batch_neg_item):
		u = self.user_emb(batch_user)			
		i = self.item_emb(batch_pos_item)		
		j = self.item_emb(batch_neg_item)	
		
		pos_dist = ((u - i) ** 2).sum(dim=1, keepdim=True)
		neg_dist = ((u - j) ** 2).sum(dim=1, keepdim=True)

		output = (pos_dist, neg_dist)

		return output

	def get_loss(self, output):
		pos_dist, neg_dist = output[0], output[1]
		loss = F.relu(self.margin + pos_dist - neg_dist).sum()
		
		return loss

	def forward_pair(self, batch_user, batch_items):
		u = self.user_emb(batch_user)		# batch_size x dim
		i = self.item_emb(batch_items)		# batch_size x dim
		
		dist = ((u - i) ** 2).sum(dim=1, keepdim=True)
		
		return -dist # for ranking

	def get_embedding(self):
		users = self.user_emb(self.user_list)
		items = self.item_emb(self.item_list)

		return users, items

class VAE(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(VAE, self).__init__()
        self.num_users = num_user
        self.num_items = num_item
        self.hid_dim = emb_dim
        
        self.E = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_item, emb_dim[0]),
            nn.Tanh(),
        )
        self.E_mean = nn.Linear(emb_dim[0], emb_dim[1])
        self.E_logvar = nn.Linear(emb_dim[0], emb_dim[1])

        self.D = nn.Sequential(
            nn.Linear(emb_dim[1], emb_dim[0]),
            nn.Tanh(),
            nn.Linear(emb_dim[0], num_item),
        ) 
        
    def forward(self, u):
        h = self.E(u)
        mu = self.E_mean(h)
        logvar = self.E_logvar(h)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
                
        u_recon = self.D(z)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return u_recon, KL

    def get_loss(self, u, u_recon, KL, beta):
        nll = -torch.sum(u * nn.functional.log_softmax(u_recon))
        
        return nll + beta*KL

    def forward_pair(self, u):
        
        return None


