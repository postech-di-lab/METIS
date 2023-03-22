import numpy as np
import os
import random
import pickle
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 
import numpy as np
import copy
from pdb import set_trace as bp

def Euclidian_dist(user_mat, item_mat):
	A = (user_mat ** 2).sum(1, keepdim=True)
	B = (item_mat ** 2).sum(1, keepdim=True)
	
	AB = -2 * torch.matmul(user_mat, item_mat.T)
	
	return torch.sqrt(A + AB + B.T)   

def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, user_id, item_id, val):
	if user_id in base_dict:
		base_dict[user_id][item_id] = val
	else:
		base_dict[user_id] = {item_id: val}


def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False


def save_pickle(path, filename, obj):
	with open(path + filename, 'wb') as f:
		pickle.dump(obj, f)


def load_pickle(path, filename):
	with open(path + filename, 'rb') as f:
		obj = pickle.load(f)

	return obj


def list_to_dict(base_list):
	result = {}
	for user_id, item_id, value in base_list:
		dict_set(result, user_id, item_id, value)
	
	return result


def dict_to_list(base_dict):
	result = []

	for user_id in base_dict:
		for item_id in base_dict[user_id]:
			result.append((user_id, item_id, 1))
	
	return result
	
	
def read_data(f):
	
	total_interactions = []
	user_count, item_count = 0, 0

	for user_id, line in enumerate(f.readlines()):
		items = line.split(' ')[1:]
		
		user_count = max(user_count, user_id)
		for item in items:
			item_id = int(item)
			item_count = max(item_count, item_id)
			
			total_interactions.append((user_id, item_id, 1))

	return user_count + 1, item_count + 1, total_interactions



def get_count_dict(total_interactions, spliter="\t"):

	user_count_dict, item_count_dict = {}, {}

	for line in total_interactions:
		user, item, rating = line
		user, item, rating = int(user), int(item), float(rating)

		if user in user_count_dict:
			user_count_dict[user] += 1
		else: 
			user_count_dict[user] = 1

		if item in item_count_dict:
			item_count_dict[item] += 1
		else: 
			item_count_dict[item] = 1

	return user_count_dict, item_count_dict


def get_total_interactions(total_interaction_tmp, user_count_dict, item_count_dict, is_implicit=True, spliter="\t"):

	total_interactions = []
	train_interactions = []
	user_dict, item_dict = {}, {}
	user_count, item_count = 0, 0

	for line in total_interaction_tmp:

		train_only = False
		user, item, rating = line
		user, item, rating = int(user), int(item), float(rating)

		# Here, we apply 10-core filtering for fast test
		if user_count_dict[user] < 10:
			continue
		if item_count_dict[item] < 10:
			continue

		if user in user_dict:
			user_id = user_dict[user]
		else:
			user_id = user_count
			user_dict[user] = user_id
			user_count += 1

		if item in item_dict:
			item_id = item_dict[item]
		else:
			item_id = item_count
			item_dict[item] = item_id
			item_count += 1

		if train_only:
			train_interactions.append((user_id, item_id, 1))
		else:
			total_interactions.append((user_id, item_id, 1))

	return user_count + 1, item_count + 1, total_interactions, train_interactions


def load_data():
	
	np.random.seed(0)
	test_ratio = 0.2
	
	with open('./Dataset/C/users.dat', 'r') as f:
		user_count, item_count, total_interaction_tmp = read_data(f)

	user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
	user_count, item_count, total_interactions, train_interactions = get_total_interactions(total_interaction_tmp, user_count_dict, item_count_dict) 

	total_mat = list_to_dict(total_interactions)
	
	train_mat, valid_mat, test_mat = {}, {}, {}

	for user in total_mat:
		items = list(total_mat[user].keys())
		np.random.shuffle(items)

		num_test_items = int(len(items) * test_ratio)
		test_items = items[:num_test_items]
		valid_items = items[num_test_items: num_test_items*2]
		train_items = items[num_test_items*2:]

		for item in test_items:
			dict_set(test_mat, user, item, 1)

		for item in valid_items:
			dict_set(valid_mat, user, item, 1)

		for item in train_items:
			dict_set(train_mat, user, item, 1)

	for user, item, rating in train_interactions:
		dict_set(train_mat, user, item, 1)
			
	train_mat_R = {}

	for user in train_mat:
		for item in train_mat[user]:
			dict_set(train_mat_R, item, user, 1)
			
	for user in list(valid_mat.keys()):
		for item in list(valid_mat[user].keys()):
			if item not in train_mat_R:
				del valid_mat[user][item]

		if len(valid_mat[user]) == 0:
			del valid_mat[user]
			del test_mat[user]
			
	for user in list(test_mat.keys()):
		for item in list(test_mat[user].keys()):
			if item not in train_mat_R:
				del test_mat[user][item]

		if len(test_mat[user]) == 0:
			del test_mat[user]
			del valid_mat[user]
	
	train_interactions = []
	for user in train_mat:
		for item in train_mat[user]:
			train_interactions.append([user, item, 1])
	
	return user_count, item_count, train_mat, train_interactions, valid_mat, test_mat



def topN_ranking_loss(S1, S2):

	above = S1.sum(1, keepdims=True)

	below1 = S1.flip(-1).exp().cumsum(1)		
	below2 = S2.exp().sum(1, keepdims=True)		

	below = (below1 + below2).log().sum(1, keepdims=True)
	
	return -(above - below).sum()


def setup(gpu):

	w1 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)
	w2 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)
	w3 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)
	w4 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)
	w5 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)
	w6 = torch.FloatTensor([1]).to(gpu).clone().detach().requires_grad_(True)

	weight_params = [w1, w2, w3, w4, w5, w6]

	S = [0,0,0,0,0,0]	
	Queue_R = [[], [], [], [], [], []]

	return weight_params, Queue_R, S


def con_gen(Queue_R, S=None):

	t_list = []
	s_list = []
	for idx in range(len(Queue_R)):
		t = np.where(Queue_R[idx][-1] < 100, Queue_R[idx][-1], 10000)	# for fast computation
		t_list.append(t)

		# whether R or RC con
		if S != None:
			s = np.where(Queue_R[idx][-1] < 100, S[idx], 10000)			# for fast computation
			s_list.append(s/10)

	tt = 0.
	ss = 0.
	for idx in range(len(Queue_R)):
		tt += np.exp(-t_list[idx]/10)

		# whether R or RC con
		if S != None:
			ss += np.exp(-s_list[idx]/10)

	result = tt + ss

	return result


class OCCF_dataset(data.Dataset):
	def __init__(self, user_count, item_count, rating_mat, num_ns, interactions):
		super(OCCF_dataset, self).__init__()
		
		self.user_count = user_count
		self.item_count = item_count
		self.rating_mat = rating_mat
		self.num_ns = num_ns
		self.interactions = interactions
		
	def ns(self):
		
		self.train_arr = []
		sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
		
		sample_idx = 0
		for user, p_item, _ in self.interactions:
			ns_count = 0
			
			while True:
				n_item = sample_list[sample_idx]
				if not is_visited(self.rating_mat, user, n_item):
					self.train_arr.append((user, p_item, n_item))
					sample_idx += 1
					ns_count += 1
					if ns_count == self.num_ns:
						break
						
				sample_idx += 1
	
	def __len__(self):
		return len(self.interactions) * self.num_ns
		
	def __getitem__(self, idx):

		return {'user': self.train_arr[idx][0], 
				'p_item': self.train_arr[idx][1], 
				'n_item': self.train_arr[idx][2]}


class OCCF_train_dataset(OCCF_dataset):
	def __init__(self, user_count, item_count, rating_mat, interactions):
		OCCF_dataset.__init__(self, user_count, item_count, rating_mat, 1, interactions)

		self.R = torch.zeros((user_count, item_count))
		for user in rating_mat:
			items = list(rating_mat[user].keys())
			self.R[user][items] = 1.		

	def __getitem__(self, idx):

		return {'user': self.train_arr[idx][0], 
				'p_item': self.train_arr[idx][1], 
				'n_item': self.train_arr[idx][2],
				}

	def for_AE_b(self, mini_b, gpu):

		b_user = mini_b['user']
		b_pos = mini_b['p_item']
		b_neg = mini_b['n_item']

		b_user = to_np(b_user)
		b_item = to_np(torch.cat([b_pos, b_neg], 0))

		u_value, u_indices = np.unique(b_user, return_index=True)
		i_value, i_indices = np.unique(b_item, return_index=True)

		mini_b['bu'] = torch.LongTensor(u_indices).to(gpu)
		mini_b['u_vec'] = self.R[torch.LongTensor(u_value)].to(gpu)

		mini_b['bi'] = torch.LongTensor(i_indices).to(gpu)
		mini_b['i_vec'] = self.R.T[torch.LongTensor(i_value)].to(gpu)

		return mini_b


class OCCF_test_dataset(data.Dataset):
	def __init__(self, user_count, item_count, valid_mat, test_mat, b_size=512):
		super(OCCF_test_dataset, self).__init__()

		self.user_count = user_count
		self.item_count = item_count
		self.user_list = torch.LongTensor([i for i in range(user_count)])

		self.valid_mat = valid_mat
		self.test_mat = test_mat
		self.b_size = b_size

		self.b_start = 0

	def get_next_b_users(self):
		b_start = self.b_start
		b_end = self.b_start + self.b_size

		if b_end >= self.user_count:
			b_end = self.user_count
			self.b_start = 0
			return self.user_list[b_start: b_end], True
		else:
			self.b_start += self.b_size
			return self.user_list[b_start: b_end], False



def interval(tic, toc):
	print(toc-tic)


def score_mat_2_rank_mat_torch(score_mat, train_interactions, gpu=None):
	
	row, col = torch.LongTensor(np.asarray(train_interactions)[:,0]), torch.LongTensor(np.asarray(train_interactions)[:,1])
	score_mat[row, col] = score_mat.min()
	rank_tmp = torch.argsort(-score_mat, dim=-1)
	
	rank_mat = torch.zeros_like(rank_tmp).to(gpu)
	for i in range(rank_mat.shape[0]):
		row = rank_tmp[i]
		rank_mat[i][row] = torch.LongTensor(np.arange(len(row))).to(gpu)
		
	return rank_mat
	

def get_eval(train_mat, valid_mat, test_mat, sorted_mat):
	metrics = {'R50':[], 'N50':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	for test_user in test_mat:
		
		sorted_list = list(to_np(sorted_mat[test_user]))
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
				gt_mat = test_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

			for item in sorted_list:

				if item not in already_seen_items:
					sorted_list_tmp.append(item)

				if len(sorted_list_tmp) >=50: break

			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
			
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))	
			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
			
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)
			
	for mode in ['test', 'valid']:
		eval_results[mode]['R' + str(50)] = round(np.asarray(eval_results[mode]['R' + str(50)]).mean(), 4)  
		eval_results[mode]['N' + str(50)] = round(np.asarray(eval_results[mode]['N' + str(50)]).mean(), 4)   
	
	return eval_results


def get_eval_np(train_mat, valid_mat, test_mat, sorted_mat):
	metrics = {'R50':[], 'N50':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	train_mat_R = {}

	for user in train_mat:
		for item in train_mat[user]:
			dict_set(train_mat_R, item, user, 1)

	for test_user in test_mat:
		
		sorted_list = list(sorted_mat[test_user])
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
				gt_mat = test_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

			for item in sorted_list:

				if item not in already_seen_items:
					sorted_list_tmp.append(item)

				if len(sorted_list_tmp) >= 50: break
				
			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
		
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))	
			
			# ndcg
			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
			
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)
			
	for mode in ['test', 'valid']:
		eval_results[mode]['R' + str(50)] = round(np.asarray(eval_results[mode]['R' + str(50)]).mean(), 4)  
		eval_results[mode]['N' + str(50)] = round(np.asarray(eval_results[mode]['N' + str(50)]).mean(), 4)   
	
	return eval_results


def evaluate(model, gpu, train_loader, test_dict):
	test_dataset = test_dict['test_dataset']
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	H_u, H_i, _ = model.forward_encoder(model.user_list, model.item_list, model.item_list)

	A_score_mat = model.get_A_score_mat(H_u[0], H_i[0])
	A_sorted_mat = torch.argsort(A_score_mat, dim=1, descending=True)
	A_results = get_eval(train_mat, valid_mat, test_mat, A_sorted_mat)
	del A_sorted_mat
	A_rank_mat = to_np(score_mat_2_rank_mat_torch(A_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del A_score_mat

	B_score_mat = model.get_B_score_mat(H_u[1], H_i[1])
	B_sorted_mat = torch.argsort(B_score_mat, dim=1, descending=True)
	B_results = get_eval(train_mat, valid_mat, test_mat, B_sorted_mat)
	del B_sorted_mat
	B_rank_mat = to_np(score_mat_2_rank_mat_torch(B_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del B_score_mat

	C_score_mat = model.get_C_score_mat(H_u[2], H_i[2])
	C_sorted_mat = torch.argsort(C_score_mat, dim=1, descending=True)
	C_results = get_eval(train_mat, valid_mat, test_mat, C_sorted_mat)
	del C_sorted_mat
	C_rank_mat = to_np(score_mat_2_rank_mat_torch(C_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del C_score_mat

	D_score_mat = model.get_D_score_mat(H_u[3], H_i[3])
	D_sorted_mat = torch.argsort(D_score_mat, dim=1, descending=True)
	D_results = get_eval(train_mat, valid_mat, test_mat, D_sorted_mat)
	del D_sorted_mat
	D_rank_mat = to_np(score_mat_2_rank_mat_torch(D_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del D_score_mat

	E_score_mat = model.get_E_score_mat(H_u[4])
	E_sorted_mat = torch.argsort(E_score_mat, dim=1, descending=True)
	E_results = get_eval(train_mat, valid_mat, test_mat, E_sorted_mat)
	del E_sorted_mat
	E_rank_mat = to_np(score_mat_2_rank_mat_torch(E_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del E_score_mat

	sub_score_mat = model.get_sub_score_mat(H_i[5])
	sub_sorted_mat = torch.argsort(sub_score_mat, dim=1, descending=True)
	sub_results = get_eval(train_mat, valid_mat, test_mat, sub_sorted_mat)
	del sub_sorted_mat
	sub_rank_mat = to_np(score_mat_2_rank_mat_torch(sub_score_mat, train_loader.dataset.interactions, gpu=gpu))

	return (A_results, B_results, C_results, D_results, E_results, sub_results), (A_rank_mat, B_rank_mat, C_rank_mat, D_rank_mat, E_rank_mat, sub_rank_mat)



def evaluate_singleCF(model, gpu, train_loader, test_dict):
	test_dataset = test_dict['test_dataset']
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	H_u, H_i, _ = model.forward_encoder(model.user_list, model.item_list, model.item_list)

	A_score_mat = model.get_A_score_mat(H_u, H_i)
	A_sorted_mat = torch.argsort(A_score_mat, dim=1, descending=True)
	A_results = get_eval(train_mat, valid_mat, test_mat, A_sorted_mat)

	return A_results


