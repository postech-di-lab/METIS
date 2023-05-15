from data_utils import *
from Utils.dataloader import *

import numpy as np
import torch
import copy
import time


def print_result(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):

	if is_improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f} *' .format(epoch, max_epoch, train_loss, train_time, test_time))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))


	for mode in ['valid', 'test']:
		for topk in [10, 50]:
			r = eval_results[mode]['R' + str(topk)] 
			n = eval_results[mode]['N' + str(topk)] 

			print('{} R@{}: {:.4f}, N@{}: {:.4f}'.format(mode, topk, r, topk, n))

		print()


def evaluate(model, gpu, train_loader, test_dataset, return_score_mat=False, return_sorted_mat=False):
	
	eval_results = {'test': {'R50':[], 'N50':[], 'R10':[], 'N10':[]}, \
					'valid': {'R50':[], 'N50':[], 'R10':[], 'N10':[]}}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	user_emb, item_emb = model.get_embedding()
	score_mat = torch.matmul(user_emb, item_emb.T)
	sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
	score_mat = - score_mat

	sorted_mat = to_np(sorted_mat)

	for test_user in test_mat:
		
		if test_user not in train_mat: continue
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
				if len(sorted_list_tmp) == 50: break
				
			hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
		
			eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))	
			
			# ndcg
			denom = np.log2(np.arange(2, 10 + 2))
			dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
			idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])
	
			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
			
			eval_results[mode]['N10'].append(dcg_10 / idcg_10)
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)
			
	
	for mode in ['test', 'valid']:
		for topk in [50, 10]:
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   

	if return_score_mat:
		return eval_results, score_mat

	if return_sorted_mat:
		return eval_results, sorted_mat	
	return eval_results



def get_eval_result(train_mat, valid_mat, test_mat, sorted_mat):

	eval_results = {'test': {'R50':[], 'N50':[], 'R10':[], 'N10':[]}, \
					'valid': {'R50':[], 'N50':[], 'R10':[], 'N10':[]}}

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
			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
		
			eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))	
			
			# ndcg
			denom = np.log2(np.arange(2, 10 + 2))
			dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
			idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])

			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
			
			eval_results[mode]['N10'].append(dcg_10 / idcg_10)
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)
			
	
	# valid, test
	for mode in ['test', 'valid']:
		for topk in [50, 10]:
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   
	
	return eval_results
