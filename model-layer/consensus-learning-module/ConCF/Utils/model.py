import torch.nn.functional as F
import torch.nn as nn
import torch
from Utils.utils import *


class SingleCF(nn.Module):

	def __init__(self, user_count, item_count, dim, gpu):
		super(SingleCF, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.gpu = gpu
		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

		self.setup_network(dim)

		# for Eval
		self.test_b_start = 0
		self.test_b_size = 2048

	def get_next_b_users(self):
		b_start = self.test_b_start
		b_end = self.test_b_start + self.test_b_size

		if b_end >= self.user_count:
			b_end = self.user_count
			self.test_b_start = 0
			return self.user_list[b_start: b_end], True
		else:
			self.test_b_start += self.test_b_size
			return self.user_list[b_start: b_end], False


	def get_next_b_items(self):
		b_start = self.test_b_start
		b_end = self.test_b_start + self.test_b_size

		if b_end >= self.item_count:
			b_end = self.item_count
			self.test_b_start = 0
			return self.item_list[b_start: b_end], True
		else:
			self.test_b_start += self.test_b_size
			return self.item_list[b_start: b_end], False				


	def setup_network(self, dim):
		self.A_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.A_network = nn.Sequential(nn.Linear(dim //4, dim // 4))


	def A_head(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		A_u = self.A_network(h_u)
		A_i = self.A_network(h_i)
		A_j = self.A_network(h_j)
		
		A_pos_score = (A_u * A_i).sum(dim=1, keepdim=True)
		A_neg_score = (A_u * A_j).sum(dim=1, keepdim=True)

		A_loss = -(A_pos_score - A_neg_score).sigmoid().log().sum()

		return A_loss		


	def forward_encoder(self, user, p_item, n_item):

		H_u = self.A_bottom_network(self.user_emb(user))

		H_i = self.A_bottom_network(self.item_emb(p_item))

		H_j = self.A_bottom_network(self.item_emb(n_item))

		return H_u, H_i, H_j


	def forward(self, mini_b):

		user = mini_b['user']
		p_item = mini_b['p_item']
		n_item = mini_b['n_item']

		user_indices_AE = mini_b['bu']
		u_vec_AE = mini_b['u_vec']

		output = self.forward_encoder(user, p_item, n_item)

		return output

		
	def get_loss(self, output):

		H_u, H_i, H_j = output[0], output[1], output[2]
		A_loss = self.A_head((H_u, H_i, H_j))

		return A_loss


	def get_A_score_mat(self, h_u, h_i):
		user_emb = self.A_network(h_u)
		item_emb = self.A_network(h_i)		
		score_mat = torch.matmul(user_emb, item_emb.T)

		return score_mat


# Multi-Branch Architecture
class MBA(nn.Module):

	def __init__(self, user_count, item_count, dim, gpu):
		super(MBA, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.gpu = gpu
		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

		self.setup_network(dim)

		self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')
		self.MSE_loss = nn.MSELoss(reduction='sum')

		# for Eval
		self.test_b_start = 0
		self.test_b_size = 2048

	def get_next_b_users(self):
		b_start = self.test_b_start
		b_end = self.test_b_start + self.test_b_size

		if b_end >= self.user_count:
			b_end = self.user_count
			self.test_b_start = 0
			return self.user_list[b_start: b_end], True
		else:
			self.test_b_start += self.test_b_size
			return self.user_list[b_start: b_end], False

	def get_next_b_items(self):
		b_start = self.test_b_start
		b_end = self.test_b_start + self.test_b_size

		if b_end >= self.item_count:
			b_end = self.item_count
			self.test_b_start = 0
			return self.item_list[b_start: b_end], True
		else:
			self.test_b_start += self.test_b_size
			return self.item_list[b_start: b_end], False				


	def setup_network(self, dim):
		self.A_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.B_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.C_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.E_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.D_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.sub_bottom_network = nn.Sequential(nn.Linear(dim, dim // 2), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(dim // 2, dim // 4))
		self.sub_network = nn.Sequential(nn.Linear(dim //4, self.user_count))

		self.A_network = nn.Sequential(nn.Linear(dim //4, dim // 4))
		self.B_network = nn.Sequential(nn.Linear(dim //4, dim // 4))
		self.C_network = nn.Sequential(nn.Linear(dim //4, 1))
		self.E_network = nn.Sequential(nn.Linear(dim //4, self.item_count))
		self.D_network = nn.Sequential(nn.Linear(dim //4, 1))


	def A_head(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		A_u = self.A_network(h_u)
		A_i = self.A_network(h_i)
		A_j = self.A_network(h_j)
		
		A_pos_score = (A_u * A_i).sum(dim=1, keepdim=True)
		A_neg_score = (A_u * A_j).sum(dim=1, keepdim=True)

		A_loss = -(A_pos_score - A_neg_score).sigmoid().log().sum()

		return A_loss		


	def B_head(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		B_u = self.B_network(h_u)
		B_i = self.B_network(h_i)
		B_j = self.B_network(h_j)

		B_u = F.normalize(B_u, dim=-1, eps=1.)
		B_i = F.normalize(B_i, dim=-1, eps=1.)
		B_j = F.normalize(B_j, dim=-1, eps=1.)

		pos_dist = ((B_u - B_i) ** 2).sum(dim=1, keepdim=True)
		neg_dist = ((B_u - B_j) ** 2).sum(dim=1, keepdim=True)
		
		B_loss = F.relu(1.0 + pos_dist - neg_dist).sum()

		return B_loss
	

	def C_head(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		C_pos = self.C_network(h_u * h_i)
		C_neg = self.C_network(h_u * h_j)
		
		pred = torch.cat([C_pos, C_neg], dim=0)
		gt = torch.cat([torch.ones_like(C_pos), torch.zeros_like(C_neg)], dim=0)
		C_loss = self.BCE_loss(pred, gt)

		return C_loss


	def D_head(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		D_pos = self.D_network(h_u * h_i)
		D_neg = self.D_network(h_u * h_j)
		
		pred = torch.cat([D_pos, D_neg], dim=0)
		gt = torch.cat([torch.ones_like(D_pos), torch.zeros_like(D_neg)], dim=0)
		D_loss = self.MSE_loss(pred, gt)

		return D_loss


	def E_head(self, output):

		h_u, rating_vec = output[0], output[1]
		output_u = self.E_network(h_u)
		output_u = F.softmax(output_u, dim=-1)

		E_loss = - torch.log(output_u + 1e-12) * rating_vec
		E_loss = torch.sum(torch.sum(E_loss, dim=1), dim=0)		

		return E_loss


	def sub_head(self, output):

		h_i, i_vec = output[0], output[1]
		output_i = self.sub_network(h_i)
		output_i = F.softmax(output_i, dim=-1)

		sub_loss = - torch.log(output_i + 1e-12) * i_vec
		sub_loss = torch.sum(torch.sum(sub_loss, dim=1), dim=0)		

		return sub_loss


	def forward_encoder(self, user, p_item, n_item):

		H_u = [self.A_bottom_network(self.user_emb(user)), self.B_bottom_network(self.user_emb(user)), self.C_bottom_network(self.user_emb(user)), \
		self.D_bottom_network(self.user_emb(user)), self.E_bottom_network(self.user_emb(user)), self.sub_bottom_network(self.user_emb(user))]

		H_i = [self.A_bottom_network(self.item_emb(p_item)), self.B_bottom_network(self.item_emb(p_item)), self.C_bottom_network(self.item_emb(p_item)), \
		self.D_bottom_network(self.item_emb(p_item)), self.E_bottom_network(self.item_emb(p_item)), self.sub_bottom_network(self.item_emb(p_item))]

		H_j = [self.A_bottom_network(self.item_emb(n_item)), self.B_bottom_network(self.item_emb(n_item)), self.C_bottom_network(self.item_emb(n_item)), \
		self.D_bottom_network(self.item_emb(n_item)), self.E_bottom_network(self.item_emb(n_item)), self.sub_bottom_network(self.item_emb(n_item))]

		return H_u, H_i, H_j


	def forward(self, mini_b):

		user = mini_b['user']
		p_item = mini_b['p_item']
		n_item = mini_b['n_item']

		user_indices_AE = mini_b['bu']
		u_vec_AE = mini_b['u_vec']

		H_u, H_i, H_j = self.forward_encoder(user, p_item, n_item)

		item_indices_AE = mini_b['bi']
		i_vec_AE = mini_b['i_vec']		

		return 	H_u[:4], H_i[:4], H_j[:4], \
				H_u[4][user_indices_AE], u_vec_AE, \
				torch.cat([H_i[5], H_j[5]],0)[item_indices_AE], i_vec_AE

		
	def get_loss(self, output, weight_params=None):

		H_u, H_i, H_j = output[0], output[1], output[2]
		h_u_E, u_vec_AE = output[3], output[4]
		h_i_sub, i_vec_AE = output[5], output[6]
	
		A_loss = self.A_head((H_u[0], H_i[0], H_j[0]))
		B_loss = self.B_head((H_u[1], H_i[1], H_j[1]))
		C_loss = self.C_head((H_u[2], H_i[2], H_j[2]))
		D_loss = self.D_head((H_u[3], H_i[3], H_j[3]))
		E_loss = self.E_head((h_u_E, u_vec_AE))
		sub_loss = self.sub_head((h_i_sub, i_vec_AE))	


		return A_loss * weight_params[0], B_loss * weight_params[1] \
				,C_loss * weight_params[2], D_loss * weight_params[3] \
				, E_loss * weight_params[4], sub_loss * weight_params[5]

	def get_A_score_mat(self, h_u, h_i):
		user_emb = self.A_network(h_u)
		item_emb = self.A_network(h_i)		
		score_mat = torch.matmul(user_emb, item_emb.T)

		return score_mat

	def get_B_score_mat(self, h_u, h_i):
		user_emb = self.B_network(h_u)
		item_emb = self.B_network(h_i)

		user_emb = F.normalize(user_emb, dim=-1, eps=1.)
		item_emb = F.normalize(item_emb, dim=-1, eps=1.)

		score_mat = Euclidian_dist(user_emb, item_emb)

		return -score_mat

	def get_C_score_mat(self, h_u, h_i):
		score_mat = torch.zeros(self.user_count, self.item_count).to(self.gpu)

		while True:
			b_users_org, is_last_b = self.get_next_b_users()
			total_items = torch.cat(b_users_org.size(0) * [self.item_list.unsqueeze(0)], 0)

			b_users = b_users_org.unsqueeze(-1)
			b_users = torch.cat(total_items.size(1) * [b_users], 1).to(self.gpu)

			score_mat_tmp = self.C_network(h_u[b_users] * h_i[total_items]).sigmoid().squeeze(-1)

			score_mat[b_users_org, :] = score_mat_tmp

			if is_last_b:
				break

		return score_mat

	def get_D_score_mat(self, h_u, h_i):
		score_mat = torch.zeros(self.user_count, self.item_count).to(self.gpu)

		while True:
			b_users_org, is_last_b = self.get_next_b_users()
			total_items = torch.cat(b_users_org.size(0) * [self.item_list.unsqueeze(0)], 0)

			b_users = b_users_org.unsqueeze(-1)
			b_users = torch.cat(total_items.size(1) * [b_users], 1).to(self.gpu)

			score_mat_tmp = self.D_network(h_u[b_users] * h_i[total_items]).squeeze(-1)

			score_mat[b_users_org, :] = score_mat_tmp

			if is_last_b:
				break

		return score_mat


	def get_E_score_mat(self, h_u):
		score_mat = torch.zeros(self.user_count, self.item_count).to(self.gpu)
		while True:
			b_users, is_last_b = self.get_next_b_users()		
			b_users = b_users.to(self.gpu)

			score_mat_tmp = self.E_network(h_u[b_users])
			score_mat_tmp = F.softmax(score_mat_tmp, dim=-1)

			score_mat[b_users, :] = score_mat_tmp		
			if is_last_b:
				break
		return score_mat


	def get_sub_score_mat(self, h_i):
		score_mat = torch.zeros(self.item_count, self.user_count).to(self.gpu)
		while True:
			b_items, is_last_b = self.get_next_b_items()		
			b_items = b_items.to(self.gpu)

			score_mat_tmp = self.sub_network(h_i[b_items])
			score_mat_tmp = F.softmax(score_mat_tmp, dim=-1)

			score_mat[b_items, :] = score_mat_tmp		
			if is_last_b:
				break

		score_mat = score_mat.T
		return score_mat


class ConCF(MBA):

	def __init__(self, user_count, item_count, dim, gpu):
		MBA.__init__(self, user_count, item_count, dim, gpu)
		
	# CL loss
	def CL_loss(self, b_user, CL_item, weight_params):

		CL_u, CL_i = b_user.unique(), CL_item.view((-1,))
		CL_u_unique, CL_u_unique_indices = torch.unique(CL_u, return_inverse=True)
		CL_i_unique, CL_i_unique_indices = torch.unique(CL_i, return_inverse=True)

		H_u, H_i, _ = self.forward_encoder(CL_u_unique, CL_i_unique, CL_i_unique)

		user_indices = torch.cat([CL_u_unique_indices.unsqueeze(-1)] * CL_item.size(1), 1)
		item_indices = CL_i_unique_indices.view((CL_u.size(0), -1))

		A_u, A_i = self.A_network(H_u[0]), self.A_network(H_i[0])
		A_CL = (A_u[user_indices] * A_i[item_indices]).sum(-1)
		A_CL = torch.clamp(A_CL, min=-40, max=40)

		A_CL1, A_CL2 = A_CL[:,:A_CL.size(1)//2], A_CL[:,A_CL.size(1)//2:]
		A_loss = topN_ranking_loss(A_CL1, A_CL2)

		B_u, B_i = self.B_network(H_u[1]), self.B_network(H_i[1])
		B_u = F.normalize(B_u, dim=-1, eps=1.)
		B_i = F.normalize(B_i, dim=-1, eps=1.)
		B_CL = -10 * ((B_u[user_indices] - B_i[item_indices]) ** 2).sum(-1)
		B_CL1, B_CL2 = B_CL[:,:A_CL.size(1)//2], B_CL[:,A_CL.size(1)//2:]
		B_loss = topN_ranking_loss(B_CL1, B_CL2)

		C_CL = (self.C_network(H_u[2][user_indices] * H_i[2][item_indices]).squeeze(-1))	
		C_CL1, C_CL2 = C_CL[:,:A_CL.size(1)//2], C_CL[:,A_CL.size(1)//2:]
		C_loss = topN_ranking_loss(C_CL1, C_CL2)

		D_CL = (self.D_network(H_u[3][user_indices] * H_i[3][item_indices]).squeeze(-1))	
		D_CL1, D_CL2 = D_CL[:,:A_CL.size(1)//2], D_CL[:,A_CL.size(1)//2:]
		D_loss = topN_ranking_loss(D_CL1, D_CL2) 

		E_CL = torch.gather(self.E_network(H_u[4]), 1, CL_item)
		E_CL = torch.clamp(E_CL, min=-40, max=40)
		E_CL1, E_CL2 = E_CL[:,:A_CL.size(1)//2], E_CL[:,A_CL.size(1)//2:]
		E_loss = topN_ranking_loss(E_CL1, E_CL2)

		# balncing clipling for stable training
		ws = [max(float(x), 0.5) for x in weight_params]
		ws = [min(float(x), 2.) for x in ws]

		return A_loss * ws[0], B_loss * ws[1], C_loss * ws[2], D_loss * ws[3], E_loss * ws[4]


				
