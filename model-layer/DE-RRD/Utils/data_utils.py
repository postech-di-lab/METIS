import numpy as np
import os
import random
import pickle
import time
import torch


########################################################################################################################
# Helper Functions
########################################################################################################################

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
	

def turn_it_to_boolean(yes_or_no):
	if yes_or_no == 'yes':
		return True
	elif yes_or_no == 'no':
		return False
	else:
		assert False


def T_annealing(epoch, max_epoch, initial_T, end_T):
	new_T = initial_T * ((end_T / initial_T) ** (epoch / max_epoch))
	return new_T


########################################################################################################################
# For data load
########################################################################################################################


def get_count_dict(f, spliter="\t"):

	user_count_dict, item_count_dict = {}, {}

	for line in f.readlines():
		tmp = line.split(spliter)

		if len(tmp) == 4:
			user, item, rating, timestamp = tmp
		else:
			user, item, rating = tmp

		if user in user_count_dict:
			user_count_dict[user] += 1
		else: 
			user_count_dict[user] = 1

		if item in item_count_dict:
			item_count_dict[item] += 1
		else: 
			item_count_dict[item] = 1

	return user_count_dict, item_count_dict


def read_LOO_settings(path, dataset, seed):

	train_mat = load_pickle(path + dataset, "/LOO/train_mat_" + str(seed))
	train_interactions = dict_to_list(train_mat)

	test_sample = load_pickle(path + dataset, "/LOO/test_sample_" + str(seed))
	valid_sample = load_pickle(path + dataset, "/LOO/valid_sample_" + str(seed))
	candidates = load_pickle(path + dataset, "/LOO/candidates_" + str(seed))

	user_count, item_count = load_pickle(path + dataset, "/LOO/counts")		

	return user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates

