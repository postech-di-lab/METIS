import numpy as np
import os
import random
import pickle
import time
import torch
import copy
from Utils.evaluation import *

## helper functions
def load_pickle(path, filename):
	with open(path + filename, 'rb') as f:
		obj = pickle.load(f)

	return obj

def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, u_id, i_id, val):
	if u_id in base_dict:
		base_dict[u_id][i_id] = val
	else:
		base_dict[u_id] = {i_id: val}


def is_visited(base_dict, u_id, i_id):
	if u_id in base_dict and i_id in base_dict[u_id]:
		return True
	else:
		return False


def list_to_dict(base_list):
	result = {}
	for u_id, i_id, value in base_list:
		dict_set(result, u_id, i_id, value)
	
	return result


def dict_to_list(base_dict):
	result = []

	for u_id in base_dict:
		for i_id in base_dict[u_id]:
			result.append((u_id, i_id, 1))
	
	return result
	
## for data load
def read_file(f):

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



def load_data(test_ratio=0.2, random_seed=0):
	
	np.random.seed(random_seed)

	with open('Data/users.dat') as f:
		u_count, i_count, total_int_tmp = read_file(f)

	u_count_dict, i_count_dict = get_count_dict(total_int_tmp)
	u_count, i_count, total_ints = get_total_ints(total_int_tmp, u_count_dict, i_count_dict, count_filtering = [5, 0])
	total_mat = list_to_dict(total_ints)

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
			
	train_mat_R = {}

	for u in train_mat:
		for i in train_mat[u]:
			dict_set(train_mat_R, i, u, 1)
			
	for u in list(valid_mat.keys()):
		for i in list(valid_mat[u].keys()):
			if i not in train_mat_R:
				del valid_mat[u][i]
		if len(valid_mat[u]) == 0:
			del valid_mat[u]
			del test_mat[u]
			
	for u in list(test_mat.keys()):
		for i in list(test_mat[u].keys()):
			if i not in train_mat_R:
				del test_mat[u][i]
		if len(test_mat[u]) == 0:
			del test_mat[u]
			del valid_mat[u]
	
	train_ints = []
	for u in train_mat:
		for i in train_mat[u]:
			train_ints.append([u, i, 1])
			
	return u_count, i_count, train_mat, train_ints, valid_mat, test_mat


def get_count_dict(total_ints, spliter="\t"):

	u_count_dict, i_count_dict = {}, {}

	for line in total_ints:
		u, i, rating = line
		u, i, rating = int(u), int(i), float(rating)

		if u in u_count_dict:
			u_count_dict[u] += 1
		else: 
			u_count_dict[u] = 1

		if i in i_count_dict:
			i_count_dict[i] += 1
		else: 
			i_count_dict[i] = 1

	return u_count_dict, i_count_dict


def get_total_ints(total_int_tmp, u_count_dict, i_count_dict, is_implicit=True, count_filtering = [10, 10], spliter="\t"):

	total_ints = []
	u_dict, i_dict = {}, {}
	u_count, i_count = 0, 0

	for line in total_int_tmp:
		u, i, rating = line
		u, i, rating = int(u), int(i), float(rating)

		# count filtering
		if u_count_dict[u] < count_filtering[0]:
			continue
		if i_count_dict[i] < count_filtering[1]:
			continue

		# u indexing
		if u in u_dict:
			u_id = u_dict[u]
		else:
			u_id = u_count
			u_dict[u] = u_id
			u_count += 1

		# i indexing
		if i in i_dict:
			i_id = i_dict[i]
		else:
			i_id = i_count
			i_dict[i] = i_id
			i_count += 1

		if is_implicit:
			rating = 1.

		total_ints.append((u_id, i_id, rating))

	return u_count + 1, i_count + 1, total_ints


def load_teacher_trajectory(path, model_list, train_interactions, K, gpu):

	# we load the precomputed importance for ensemble (Eq.8)
	state_dict = load_pickle(path, 'state_dict')
	perm_dict = load_pickle(path, 'top_perms')

	# initial top/pos permutations
	p_results = load_pickle(path, 'observed')
	hidden_positives = load_pickle(path, 'hidden_positives')

	exception_ints = []
	for u in range(hidden_positives.shape[0]):
		for i in hidden_positives[u]:
			exception_ints.append((u, i, 1))

	sorted_mat = g_torch(state_dict, torch.zeros((state_dict['MF'][0].shape[0], 6)), train_interactions, gpu)
	t_results = sorted_mat[:, :K]   
	p_results = p_results[:, :K//10]
	p_results = torch.LongTensor(p_results).to(gpu)  

	return state_dict, perm_dict, t_results, p_results, exception_ints


#############
def g(importance_mats):
	
	result = 0
	for importance_mat in importance_mats:
		result += importance_mat

	return result

def g_torch(state_dict, v, train_interactions, gpu):
	
	importance_mats = []
	
	for model_idx, model_type in enumerate(state_dict):
		importance_mat = torch.zeros(state_dict[model_type][0].shape)
	
		for user in range(v.shape[0]):
			importance_mat[user] = torch.FloatTensor(state_dict[model_type][int(v[user][model_idx])][user])
		importance_mats.append(importance_mat.to(gpu))
	
	result = g(importance_mats)
	row, col = torch.LongTensor(train_interactions)[:,0].to(gpu), torch.LongTensor(train_interactions)[:,1].to(gpu)
	result[row, col] = result.min()

	sorted_mat = torch.argsort(-result, axis=-1)
 
	return sorted_mat
