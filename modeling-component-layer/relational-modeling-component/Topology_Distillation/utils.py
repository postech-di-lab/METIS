import numpy as np
import os
import random
import pickle
import time
import torch
import math
import copy 
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 


class train_dataset(data.Dataset):

	def __init__(self, user_num, item_num, rating_mat, interactions, num_ns=1):
		super(train_dataset, self).__init__()
		
		self.user_num = user_num
		self.item_num = item_num
		self.rating_mat = rating_mat
		self.num_ns = num_ns
		self.interactions = interactions
		

	def negative_sampling(self):

		self.train_arr = []
		sample_list = np.random.choice(list(range(self.item_num)), size = 10 * len(self.interactions) * self.num_ns)
		
		sample_idx = 0
		for user, pos_item, _ in self.interactions:
			ns_count = 0
			
			while True:
				neg_item = sample_list[sample_idx]
				if not is_visited(self.rating_mat, user, neg_item):
					self.train_arr.append((user, pos_item, neg_item))
					sample_idx += 1
					ns_count += 1
					if ns_count == self.num_ns:
						break
						
				sample_idx += 1

	def __len__(self):
		return len(self.interactions) * self.num_ns
		

	def __getitem__(self, idx):
		return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]



class test_dataset(data.Dataset):
	def __init__(self, user_num, item_num, valid_mat, test_mat):
		super(test_dataset, self).__init__()

		self.user_num = user_num
		self.item_num = item_num
		self.user_list = torch.LongTensor([i for i in range(user_num)])

		self.valid_mat = valid_mat
		self.test_mat = test_mat


def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False
		
def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, user_id, item_id, val):
	if user_id in base_dict:
		base_dict[user_id][item_id] = val
	else:
		base_dict[user_id] = {item_id: val}


def dict_2_list(base_dict):
	result = []

	for user_id in base_dict:
		for item_id in base_dict[user_id]:
			result.append((user_id, item_id, 1))
	
	return result
	

def sim(A, B, is_inner=False):
	
	if not is_inner:
		denom_A = 1 / (A ** 2).sum(1, keepdim=True).sqrt()
		denom_B = 1 / (B.T ** 2).sum(0, keepdim=True).sqrt()

		sim_mat = torch.mm(A, B.T) * denom_A * denom_B
	else:
		sim_mat = torch.mm(A, B.T)

	return sim_mat
	

def read_settings():

	with open("./dataset/Train", 'rb') as f:
		Train = pickle.load(f)

	with open("./dataset/Test", 'rb') as f:
		Test = pickle.load(f)

	with open("./dataset/Valid", 'rb') as f:
		Valid = pickle.load(f)
		
	train_interactions = dict_2_list(Train)

	valid_mat = {}
	test_mat = {}

	for user in Valid:
		item = Valid[user]
		dict_set(valid_mat, user, item, 1)

	for user in Test:
		item = Test[user]
		dict_set(test_mat, user, item, 1)

	return Train, train_interactions, valid_mat, test_mat


def print_result(epoch, max_epoch, train_loss, eval_results):

	print('Epoch [{}/{}], Train Loss: {:.4f}' .format(epoch, max_epoch, train_loss))


	for mode in ['valid', 'test']:
		for topk in [10, 20, 50]:
			p = eval_results[mode]['P' + str(topk)]
			r = eval_results[mode]['R' + str(topk)] 
			n = eval_results[mode]['N' + str(topk)] 

			print('{} Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(mode, topk, r, topk, n))

		print()


def evaluate(model, gpu, train_loader, test_dataset):
	
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	user_emb, item_emb = model.get_embedding()
	score_mat = torch.matmul(user_emb, item_emb.T)
	sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
	
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
				if len(sorted_list_tmp) == 50: break
				
			hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
			hit_20 = len(set(sorted_list_tmp[:20]) & set(gt_mat[test_user].keys()))
			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
		
			eval_results[mode]['P10'].append(hit_10 / min(10, len(gt_mat[test_user].keys())))
			eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))
			
			eval_results[mode]['P20'].append(hit_20 / min(20, len(gt_mat[test_user].keys())))
			eval_results[mode]['R20'].append(hit_20 / len(gt_mat[test_user].keys()))
			
			eval_results[mode]['P50'].append(hit_50 / min(50, len(gt_mat[test_user].keys())))
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))	
			
			# ndcg
			denom = np.log2(np.arange(2, 10 + 2))
			dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
			idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])
			
			denom = np.log2(np.arange(2, 20 + 2))
			dcg_20 = np.sum(np.in1d(sorted_list_tmp[:20], list(gt_mat[test_user].keys())) / denom)
			idcg_20 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 20)])
			
			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
			
			eval_results[mode]['N10'].append(dcg_10 / idcg_10)
			eval_results[mode]['N20'].append(dcg_20 / idcg_20)
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)


	# valid, test
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   

	return eval_results

