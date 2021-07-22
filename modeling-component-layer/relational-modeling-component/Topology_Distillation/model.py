import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import sim, to_np
import math
from pdb import set_trace as bp

class f(nn.Module):
	def __init__(self, dims):
		super(f, self).__init__()

		self.net = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]))

	def forward(self, x):
		return self.net(x)


class Base_model(nn.Module):
	def __init__(self, user_num, item_num, dim, gpu):
		super(Base_model, self).__init__()
		self.user_num = user_num
		self.item_num = item_num

		self.user_list = torch.LongTensor([i for i in range(user_num)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_num)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_num, dim)
		self.item_emb = nn.Embedding(self.item_num, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
		
	def forward(self, batch_user, batch_pos_item, batch_neg_item):
		u = self.user_emb(batch_user)			
		i = self.item_emb(batch_pos_item)		
		j = self.item_emb(batch_neg_item)		
		
		pos_score = (u * i).sum(dim=1, keepdim=True)
		neg_score = (u * j).sum(dim=1, keepdim=True)

		output = (pos_score, neg_score)

		return output

	def get_loss(self, output):
		pos_score, neg_score = output[0], output[1]
		loss = -(pos_score - neg_score).sigmoid().log().sum()
		
		return loss

	def get_embedding(self):
		users = self.user_emb(self.user_list)
		items = self.item_emb(self.item_list)

		return users, items


class FTD(Base_model):
	def __init__(self, user_num, item_num, user_emb_teacher, item_emb_teacher, gpu, student_dim):

		Base_model.__init__(self, user_num, item_num, student_dim, gpu)

		self.student_dim = student_dim
		self.gpu = gpu

		# Teacher
		self.user_emb_teacher = nn.Embedding.from_pretrained(user_emb_teacher)
		self.item_emb_teacher = nn.Embedding.from_pretrained(item_emb_teacher)

		self.user_emb_teacher.weight.requires_grad = False
		self.item_emb_teacher.weight.requires_grad = False

		self.teacher_dim = self.user_emb_teacher.weight.size(1)

	# topology distillation loss
	def get_TD_loss(self, batch_user, batch_item):

		s = torch.cat([self.user_emb(batch_user), self.item_emb(batch_item)], 0)
		t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0)

		# Full topology
		t_dist = sim(t, t).view(-1)
		s_dist = sim(s, s).view(-1)  

		total_loss = ((t_dist - s_dist) ** 2).sum() 

		return total_loss


class HTD(Base_model):
	def __init__(self, user_num, item_num, user_emb_teacher, item_emb_teacher, gpu, student_dim, K, choice):

		Base_model.__init__(self, user_num, item_num, student_dim, gpu)

		self.student_dim = student_dim
		self.gpu = gpu

		# Teacher
		self.user_emb_teacher = nn.Embedding.from_pretrained(user_emb_teacher)
		self.item_emb_teacher = nn.Embedding.from_pretrained(item_emb_teacher)

		self.user_emb_teacher.weight.requires_grad = False
		self.item_emb_teacher.weight.requires_grad = False

		self.teacher_dim = self.user_emb_teacher.weight.size(1)

		# Group Assignment related parameters
		self.K = K
		F_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]

		self.user_f = nn.ModuleList([f(F_dims) for i in range(self.K)])
		self.item_f = nn.ModuleList([f(F_dims) for i in range(self.K)])

		self.user_v = nn.Sequential(nn.Linear(self.teacher_dim, K), nn.Softmax(dim=1))
		self.item_v = nn.Sequential(nn.Linear(self.teacher_dim, K), nn.Softmax(dim=1))

		self.sm = nn.Softmax(dim = 1)	
		self.T = 0.1

		# Group-Level topology design choices
		self.choice = choice


	def get_group_result(self, batch_entity, is_user=True):
		with torch.no_grad():
			if is_user:
				t = self.user_emb_teacher(batch_entity)		
				v = self.user_v
			else:
				t = self.item_emb_teacher(batch_entity)		
				v = self.item_v

			z = v(t).max(-1)[1] 
			if not is_user:
				z = z + self.K
				
			return z


	# For Adaptive Group Assignment
	def get_GA_loss(self, batch_entity, is_user=True):

		if is_user:
			s = self.user_emb(batch_entity)													
			t = self.user_emb_teacher(batch_entity)										

			f = self.user_f
			v = self.user_v
		else:
			s = self.item_emb(batch_entity)													
			t = self.item_emb_teacher(batch_entity)											
	
			f = self.item_f
			v = self.item_v

		alpha = v(t) 														
		g = torch.distributions.Gumbel(0, 1).sample(alpha.size()).to(self.gpu)
		alpha = alpha + 1e-10 												
		z = self.sm((alpha.log() + g) / self.T)

		z = torch.unsqueeze(z, 1)									
		z = z.repeat(1, self.teacher_dim, 1)						

		f_hat = [f[i](s).unsqueeze(-1) for i in range(self.K)] 		
		f_hat = torch.cat(f_hat, -1)											
		f_hat = f_hat * z										
		f_hat = f_hat.sum(2)													

		GA_loss = ((t-f_hat) ** 2).sum(-1).sum() 

		return GA_loss 


	def get_TD_loss(self, batch_user, batch_item):
		if self.choice == 'first':
			return self.get_TD_loss1(batch_user, batch_item)
		else:
			return self.get_TD_loss2(batch_user, batch_item)


	# Topology Distillation Loss (with Group(P,P))
	def get_TD_loss1(self, batch_user, batch_item):

		s = torch.cat([self.user_emb(batch_user), self.item_emb(batch_item)], 0)
		t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0)
		z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
		G_set = z.unique()
		Z = F.one_hot(z).float()	

		# Compute Prototype
		with torch.no_grad():
			tmp = Z.T
			tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
			P_s = tmp.mm(s)[G_set]
			P_t = tmp.mm(t)[G_set]

		# entity_level topology
		entity_mask = Z.mm(Z.T)        
		
		t_sim_tmp = sim(t, t) * entity_mask
		t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
		
		s_sim_dist = sim(s, s) * entity_mask    
		s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
		 
		# # Group_level topology
		t_proto_dist = sim(P_t, P_t).view(-1)
		s_proto_dist = sim(P_s, P_s).view(-1)

		total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

		return total_loss


	# Topology Distillation Loss (with Group(P,e))
	def get_TD_loss2(self, batch_user, batch_item):

		s = torch.cat([self.user_emb(batch_user), self.item_emb(batch_item)], 0)
		t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0)
		z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
		G_set = z.unique()
		Z = F.one_hot(z).float()	

		# Compute Prototype
		with torch.no_grad():
			tmp = Z.T
			tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
			P_s = tmp.mm(s)[G_set]
			P_t = tmp.mm(t)[G_set]

		# entity_level topology
		entity_mask = Z.mm(Z.T)        
		
		t_sim_tmp = sim(t, t) * entity_mask
		t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
		
		s_sim_dist = sim(s, s) * entity_mask    
		s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
		 
		# # Group_level topology 
		# t_proto_dist = (sim(P_t, t) * (1 - Z.T)[G_set]).view(-1)
		# s_proto_dist = (sim(P_s, s) * (1 - Z.T)[G_set]).view(-1)

		t_proto_dist = sim(P_t, t).view(-1)
		s_proto_dist = sim(P_s, s).view(-1)

		total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

		return total_loss


