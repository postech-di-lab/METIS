import sys
import argparse
import numpy as np

from Utils.data_utils import *
from Utils.evaluation import *
from Utils.dataloader import train_dataset, test_dataset

import torch
import torch.optim as optim
import torch.utils.data as data

from Models.HetComp import HetComp_MF

def get_NDCG_u(sorted_list, teacher_t_items, user, k=50):

	with torch.no_grad():
		top_scores = np.asarray([np.exp(-t/10) for t in range(k)])
		top_scores = ((2 ** top_scores)-1)
		
		t_items = teacher_t_items[:k]

		sorted_list_tmp = []
		for item in sorted_list:
			if user in train_mat and item not in train_mat[user]:
				sorted_list_tmp.append(item)
			if len(sorted_list_tmp) == k: break  

		if user not in train_mat:
			sorted_list_tmp = sorted_list

		denom = np.log2(np.arange(2, k + 2))
		dcg_50 = np.sum((np.in1d(sorted_list_tmp[:k], list(t_items)) * top_scores) / denom)
		idcg_50 = np.sum((top_scores / denom)[:k])

		return round(dcg_50 / idcg_50, 4)

def DKC(sorted_mat, last_max_idx, last_dist, is_first, epoch, alpha=1.05):
	
	next_idx = last_max_idx[:] 
	if is_first:
		last_dist = np.ones_like(next_idx)
		for model_idx, model_type in enumerate(perm_dict):
			for user in range(user_count):
				current_selction = int(last_max_idx[model_idx][user])
				next_v = min(3, int(next_idx[model_idx][user]) + 1)

				next_perm = perm_dict[model_type][next_v][user]
				next_dist = 1 - get_NDCG_u(sorted_mat[user], next_perm, user)
				
				last_dist[model_idx][user] = next_dist

		return next_idx.T, next_idx, last_dist

	th = alpha * (0.995 ** (epoch // p))

	for model_idx, model_type in enumerate(perm_dict):
		for user in range(user_count):
			current_selction = int(last_max_idx[model_idx][user])
			next_v = min(3, int(next_idx[model_idx][user]) + 1)
			next_next_v = min(3, int(next_idx[model_idx][user]) + 2)
			
			if current_selction == 3:
				continue
			
			current_perm = perm_dict[model_type][current_selction][user]
			next_perm = perm_dict[model_type][next_v][user]
			next_next_perm = perm_dict[model_type][next_next_v][user]
			
			next_dist = 1 - get_NDCG_u(sorted_mat[user], next_perm, user)
			
			if ((last_dist[model_idx][user] / next_dist) > th) or (last_dist[model_idx][user] / next_dist) < 1:
				next_idx[model_idx][user] += 1
				next_next_dist = 1 - get_NDCG_u(sorted_mat[user], next_next_perm, user)
				last_dist[model_idx][user] = next_next_dist

	return next_idx.T, next_idx, last_dist
	


###########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--dim', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_ns', type=int, default=1)

parser.add_argument('--test_ratio', type=float, default=0.20)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1.05)
parser.add_argument('--p', type=int, default=10)

opt = parser.parse_args()

gpu = torch.device('cuda:3') 

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

alpha = opt.alpha
p = opt.p
K = 100

#############################################################################################################################
# data load
user_count, item_count, train_mat, train_interactions, valid_mat, test_mat = load_data()

# teacher trajectory needs to be located in the below directory
path = './Teachers/'
model_list = ['MF', 'ML', 'DL', 'GNN', 'AE', 'I-AE']

# load trajectory and initial supervision
state_dict, perm_dict, t_results, p_results, exception_ints = load_teacher_trajectory(path, model_list, train_interactions, K, gpu)
train_dataset = train_dataset(user_count, item_count, train_mat, 1, train_interactions, exception_ints)
test_dataset = test_dataset(user_count, item_count, valid_mat, test_mat)
train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

##############################################################################################################################
# HetComp model 
model = HetComp_MF(user_count, item_count, opt.dim, gpu)
model = model.to(gpu)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.reg)

##############################################################################################################################
# distillation

train_losses = []
b_recall = -999
b_result, f_result = -1, -1

es = 0
verbose = 10
last_dist = None
is_first = True
v_results = np.asarray([0, 0, 0, 0, 0, 0])

last_max_idx = np.zeros((len(perm_dict), user_count))
next_idx = np.clip(last_max_idx + 1, a_min=0, a_max=3)

for epoch in range(1000):

	tic1 = time.time()
	train_loader.dataset.negative_sampling()
	ep_loss = []

	for mini_batch in train_loader:

		b_u = mini_batch['u'].unique()
		
		mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

		model.train()
		output = model(mini_batch)

		b_u = torch.LongTensor(b_u).to(gpu)
		b_u_mask = train_loader.dataset.get_user_side_mask(b_u).to(gpu)
		
		t_items = torch.index_select(t_results, 0, b_u)  
		p_items = torch.index_select(p_results, 0, b_u)   
		
		if v_results.sum() < 18: 
			KD_loss = model.get_KD_loss(b_u, p_items, t_items, b_u_mask, False)
			b_loss = KD_loss * 0.01
		else:
			KD_loss = model.get_KD_loss(b_u, p_items, t_items, b_u_mask, True)
			b_loss = KD_loss * 0.005
		
		ep_loss.append(b_loss)
		optimizer.zero_grad()
		b_loss.backward()
		optimizer.step()

	ep_loss = torch.mean(torch.stack(ep_loss)).data.cpu().numpy()
	train_losses.append(ep_loss)

	toc1 = time.time()
	if epoch % verbose == 0:
		imp = False

		model.eval()
		with torch.no_grad():
			tic2 = time.time()
			e_results, sorted_mat = evaluate(model, gpu, train_loader, test_dataset, return_sorted_mat=True)
			toc2 = time.time()

			if e_results['valid']['R50'] > b_recall: 
				imp = True
				b_recall = e_results['valid']['R50']
				b_result = e_results['valid']
				f_result = e_results['test']
				es = 0						
			else:
				imp = False
				es += 1

			print_result(epoch, 1000, ep_loss, e_results, is_improved=imp, train_time=toc1-tic1, test_time=toc2-tic2)

	### DKC
	if (epoch % p == 0) and (epoch >= 10) and v_results.sum() < 18:
		
		if is_first == True:
			v, last_max_idx, last_dist = DKC(sorted_mat, last_max_idx, last_dist, True, epoch, alpha=alpha)
			is_first = False
		else:
			v, last_max_idx, last_dist = DKC(sorted_mat, last_max_idx, last_dist, False, epoch, alpha=alpha)

		t_results = g_torch(state_dict, v, train_interactions, gpu)
		t_results = t_results[:, :K]

		v_results = np.asarray([round(x, 2) for x in v.mean(0)])
		print(v_results)

	if (epoch % verbose) == 0:
		print("="* 50)

	if es >= 5:
		break