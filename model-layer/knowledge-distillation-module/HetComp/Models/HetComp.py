import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
	def __init__(self, user_count, item_count, dim, gpu):
		super(MF, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)


	def forward(self, mini_batch):

		user = mini_batch['u']
		pos_item = mini_batch['p']
		neg_item = mini_batch['n']

		u = self.user_emb(user)
		i = self.item_emb(pos_item)
		j = self.item_emb(neg_item)

		return u, i, j
		
	def get_loss(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		bpr_pos_score = (h_u * h_i).sum(dim=1, keepdim=True)
		bpr_neg_score = (h_u * h_j).sum(dim=1, keepdim=True)
		bpr_loss = -(bpr_pos_score - bpr_neg_score).sigmoid().log().sum()

		return bpr_loss

	def get_embedding(self):

		user = self.user_emb(self.user_list)
		item = self.item_emb(self.item_list)

		return user, item

	def forward_full_items(self, batch_user):
		user = self.user_emb(batch_user)
		item = self.item_emb(self.item_list) 

		return torch.matmul(user, item.T)

class HetComp_MF(MF):
	def __init__(self, user_count, item_count, dim, gpu):
		MF.__init__(self, user_count, item_count, dim, gpu)
		
	def get_batch_full_mat(self, batch_user):
		return torch.clamp(self.forward_full_items(batch_user), min=-40, max=40)
		
	def overall_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):

		tops = torch.gather(batch_full_mat, 1, torch.cat([pos_items, top_items], -1))
		tops_els = (batch_full_mat.exp() * (1-batch_user_mask)).sum(1, keepdims=True)
		els = tops_els - torch.gather(batch_full_mat, 1, top_items).exp().sum(1, keepdims=True)
		
		above = tops.view(-1, 1)
		below = torch.cat((pos_items.size(1) + top_items.size(1)) * [els], 1).view(-1, 1) + above.exp()
		below = torch.clamp(below, 1e-5).log()

		return -(above - below).sum()

	def rank_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):
		S_pos = torch.gather(batch_full_mat, 1, pos_items)
		S_top = torch.gather(batch_full_mat, 1, top_items[:,:top_items.size(1)//2])

		below2 = (batch_full_mat.exp() * (1-batch_user_mask)).sum(1, keepdims=True) - S_top.exp().sum(1, keepdims=True)
		
		above_pos = S_pos.sum(1, keepdims=True)
		above_top = S_top.sum(1, keepdims=True)
		
		below_pos = S_pos.flip(-1).exp().cumsum(1)
		below_top = S_top.flip(-1).exp().cumsum(1)
		
		below_pos = (torch.clamp(below_pos + below2, 1e-5)).log().sum(1, keepdims=True)        
		below_top = (torch.clamp(below_top + below2, 1e-5)).log().sum(1, keepdims=True)  

		pos_KD_loss = -(above_pos - below_pos).sum()

		S_top_sub = torch.gather(batch_full_mat, 1, top_items[:,:top_items.size(1)//10])
		below2_sub = (batch_full_mat.exp() * (1-batch_user_mask)).sum(1, keepdims=True) - S_top_sub.exp().sum(1, keepdims=True)
		
		above_top_sub = S_top_sub.sum(1, keepdims=True)
		below_top_sub = S_top_sub.flip(-1).exp().cumsum(1)
		below_top_sub = (torch.clamp(below_top_sub + below2_sub, 1e-5)).log().sum(1, keepdims=True)  

		top_KD_loss = - (above_top - below_top).sum() - (above_top_sub - below_top_sub).sum()

		return  pos_KD_loss + top_KD_loss / 2

	def get_KD_loss(self, batch_user, pos_items, top_items, batch_user_mask, is_final):
		batch_full_mat = self.get_batch_full_mat(batch_user)

		if not is_final:
			KD_loss = self.overall_loss(batch_full_mat, pos_items, top_items, batch_user_mask)
		else:
			KD_loss = self.rank_loss(batch_full_mat, pos_items, top_items, batch_user_mask)
		return KD_loss
