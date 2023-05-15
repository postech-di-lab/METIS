import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 
import numpy as np
from data_utils import *

class train_dataset(data.Dataset):
	def __init__(self, user_count, item_count, rating_mat, num_ns, interactions, exception_interactions=[]):
		super(train_dataset, self).__init__()
		
		self.user_count = user_count
		self.item_count = item_count
		self.rating_mat = rating_mat
		self.num_ns = num_ns
		self.interactions = interactions
		self.exception_interactions = exception_interactions

		self.R = torch.zeros((user_count, item_count))
		for user in rating_mat:
			items = list(rating_mat[user].keys())
			self.R[user][items] = 1.

		if len(exception_interactions) > 0:
			self.exception_mat = {}
			for u, i, _ in exception_interactions:
				dict_set(self.exception_mat, u, i, 1)
		
	def negative_sampling(self):
		
		self.train_arr = []
		sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
		
		sample_idx = 0
		for user, pos_item, _ in self.interactions:
			ns_count = 0
			
			while True:
				neg_item = sample_list[sample_idx]
				if len(self.exception_interactions) > 0:
					if not is_visited(self.rating_mat, user, neg_item) and not is_visited(self.exception_mat, user, neg_item) :
						self.train_arr.append((user, pos_item, neg_item))
						sample_idx += 1
						ns_count += 1
						if ns_count == self.num_ns:
							break
				else:
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

		return {'u': self.train_arr[idx][0], 
				'p': self.train_arr[idx][1], 
				'n': self.train_arr[idx][2]}

	def get_user_side_mask(self, batch_user):
		return torch.index_select(self.R, 0 , batch_user.cpu())



class test_dataset(data.Dataset):
	def __init__(self, user_count, item_count, valid_mat, test_mat, batch_size=64):
		super(test_dataset, self).__init__()

		self.user_count = user_count
		self.item_count = item_count
		self.user_list = torch.LongTensor([i for i in range(user_count)])

		self.valid_mat = valid_mat
		self.test_mat = test_mat
		self.batch_size = batch_size

		self.batch_start = 0

	def get_next_batch_users(self):
		batch_start = self.batch_start
		batch_end = self.batch_start + self.batch_size

		if batch_end >= self.user_count:
			batch_end = self.user_count
			self.batch_start = 0
			return self.user_list[batch_start: batch_end], True
		else:
			self.batch_start += self.batch_size
			return self.user_list[batch_start: batch_end], False




