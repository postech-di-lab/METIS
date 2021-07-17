import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
	def __init__(self, user_count, item_count, dim, gpu):
		"""
		Parameters
		----------
		user_count : int
		item_count : int
		dim : int
			embedding dimension
		gpu : if available
		"""
		super(BPR, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)])
		self.item_list = torch.LongTensor([i for i in range(item_count)])

		if gpu != None:
			self.user_list = self.user_list.to(gpu)
			self.item_list = self.item_list.to(gpu)

		# User / Item Embedding
		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

		# user-item similarity type
		self.sim_type = 'inner product'
		
		
	def forward(self, batch_user, batch_pos_item, batch_neg_item):
		"""
		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_pos_item : 1-D LongTensor (batch_size)
		batch_neg_item : 1-D LongTensor (batch_size)

		Returns
		-------
		output : 
			Model output to calculate its loss function
		"""
		
		u = self.user_emb(batch_user)			
		i = self.item_emb(batch_pos_item)		
		j = self.item_emb(batch_neg_item)		
		
		pos_score = (u * i).sum(dim=1, keepdim=True)
		neg_score = (u * j).sum(dim=1, keepdim=True)

		output = (pos_score, neg_score)

		return output


	def get_loss(self, output):
		"""Compute the loss function with the model output

		Parameters
		----------
		output : 
			model output (results of forward function)

		Returns
		-------
		loss : float
		"""
		pos_score, neg_score = output[0], output[1]
		loss = -(pos_score - neg_score).sigmoid().log().sum()
		
		return loss


	def forward_multi_items(self, batch_user, batch_items):
		"""forward when we have multiple items for a user

		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_items : 2-D LongTensor (batch_size x k)

		Returns
		-------
		score : 2-D FloatTensor (batch_size x k)
		"""

		batch_user = batch_user.unsqueeze(-1)
		batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
			
		u = self.user_emb(batch_user)		# batch_size x k x dim
		i = self.item_emb(batch_items)		# batch_size x k x dim
		
		score = (u * i).sum(dim=-1, keepdim=False)	
		
		return score


	def forward_multi_users(self, batch_users, batch_item):
		"""forward when we have multiple users for a item

		Parameters
		----------
		batch_users : 2-D LongTensor (batch_size x k) 
		batch_item : 1-D LongTensor (batch_size)

		Returns
		-------
		score : 2-D FloatTensor (batch_size x k)
		"""

		batch_item = batch_item.unsqueeze(-1)
		batch_item = torch.cat(batch_users.size(1) * [batch_item], 1)
			
		u = self.user_emb(batch_users)		# batch_size x k x dim
		i = self.item_emb(batch_item)		# batch_size x k x dim
		
		score = (u * i).sum(dim=-1, keepdim=False)	
		
		return score


	def get_embedding(self):
		"""get total embedding of users and items

		Returns
		-------
		users : 2-D FloatTensor (num. users x dim)
		items : 2-D FloatTensor (num. items x dim)
		"""
		users = self.user_emb(self.user_list)
		items = self.item_emb(self.item_list)

		return users, items
