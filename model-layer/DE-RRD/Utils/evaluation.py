import torch

import math
import copy 
import time

from Utils.data_utils import to_np, Euclidian_dist

import numpy as np

from pdb import set_trace as bp

def LOO_check(ranking_list, target_item, topk=10):
	"""Calculate three ranking metrics: HR, NDCG, MRR

	Parameters
	----------
	ranking_list : 1-D array
		a recommender's prediction result
	target_item : int
		ground-truth item
	topk : int, optional
		by default 10

	Returns
	-------
	HR, NDCG, MRR: float, float, float
	"""
	k = 0
	for item_id in ranking_list:
		if k == topk: return (0., 0., 0.)
		if target_item == item_id: return (1., math.log(2.) / math.log(k + 2.), 1.0 / (k + 1.))
		k += 1


def LOO_print_result(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):
	"""print Leave-one-out evaluation results

	Parameters
	----------
	epoch : int
	max_epoch : int
		maximum training epoches
	train_loss : float
	eval_results : dict
		summarizes the evaluation results
	is_improved : bool, optional
		is the result improved compared to the last best results, by default False
	train_time :float, optional
		elapsed time for training, by default 0.
	test_time : float, optional
		elapsed time for test, by default 0.
	"""

	if is_improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: train {:.2f} test {:.2f} *' .format(epoch, max_epoch, train_loss, train_time, test_time))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: train {:.2f} test {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))


	for mode in ['valid', 'test']:
		for topk in [5, 10, 20]:
			h = eval_results[mode]['H' + str(topk)]
			n = eval_results[mode]['N' + str(topk)]
			m = eval_results[mode]['M' + str(topk)]  

			print('{} H@{}: {:.4f}, N@{}: {:.4f}, M@{}: {:.4f}'.format(mode, topk, h, topk, n, topk, m))

		print()


def print_final_result(eval_dict):
	"""print final result after the training

	Parameters
	----------
	eval_dict : dict
	"""

	for mode in ['valid', 'test']:
		print(mode)

		r_dict = {'H05':0, 'M05':0, 'N05':0, 'H10':0, 'M10':0, 'N10':0, 'H20':0, 'M20':0, 'N20':0}

		if mode == 'valid':
			key = 'best_result'
		else:
			key = 'final_result'

		for topk in [5, 10, 20]:

			if topk == 5:
				topk_str = '05'
			else:
				topk_str = str(topk)

			r_dict['H' + topk_str] = eval_dict[topk][key]['H' + str(topk)]
			r_dict['M' + topk_str] = eval_dict[topk][key]['M' + str(topk)]
			r_dict['N' + topk_str] = eval_dict[topk][key]['N' + str(topk)]

		print(r_dict)



def LOO_latent_factor_evaluate(model, test_dataset):
	"""Leave-one-out evaluation for latent factor model

	Parameters
	----------
	model : Pytorch Model
	test_dataset : Pytorch Dataset

	Returns
	-------
	eval_results : dict
		summarizes the evaluation results
	"""
	
	metrics = {'H5':[], 'M5':[], 'N5':[], 'H10':[], 'M10':[], 'N10':[], 'H20':[], 'M20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	# extract score 
	if model.sim_type == 'inner product':
		user_emb, item_emb = model.get_embedding()
		score_mat = to_np(-torch.matmul(user_emb, item_emb.T))
	elif model.sim_type == 'L2 dist':
		user_emb, item_emb = model.get_embedding()
		score_mat = to_np(Euclidian_dist(user_emb, item_emb))
	else:
		assert 'Unknown sim_type'	

	test_user_list = to_np(test_dataset.user_list)
	
	# for each test user
	for test_user in test_user_list:

		test_item = [int(test_dataset.test_item[test_user][0])]
		valid_item = [int(test_dataset.valid_item[test_user][0])]
		candidates = to_np(test_dataset.candidates[test_user]).tolist()

		total_items = test_item + valid_item + candidates
		score = score_mat[test_user][total_items]

		result = np.argsort(score).flatten().tolist()
		ranking_list = np.array(total_items)[result]

		for mode in ['test', 'valid']:
			if mode == 'test':
				target_item = test_item[0]
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == valid_item[0]))
			else:
				target_item = valid_item[0]
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == test_item[0]))
		
			for topk in [5, 10, 20]:
				(h, n, m) = LOO_check(ranking_list_tmp, target_item, topk)
				eval_results[mode]['H' + str(topk)].append(h)
				eval_results[mode]['N' + str(topk)].append(n)
				eval_results[mode]['M' + str(topk)].append(m)

	# valid, test
	for mode in ['test', 'valid']:
		for topk in [5, 10, 20]:
			eval_results[mode]['H' + str(topk)] = round(np.asarray(eval_results[mode]['H' + str(topk)]).mean(), 4)
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)
			eval_results[mode]['M' + str(topk)] = round(np.asarray(eval_results[mode]['M' + str(topk)]).mean(), 4)	

	return eval_results


def LOO_Net_evaluate(model, gpu, test_dataset):
	"""Leave-one-out evaluation for deep model

	Parameters
	----------
	model : Pytorch Model
	gpu : if available
	test_dataset : Pytorch Dataset

	Returns
	-------
	eval_results : dict
		summarizes the evaluation results
	"""
	metrics = {'H10':[], 'M10':[], 'N10':[], 'H20':[], 'M20':[], 'N20':[], 'H5':[], 'M5':[], 'N5':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	# for each batch
	while True:
		batch_users, is_last_batch = test_dataset.get_next_batch_users()
		batch_test_items, batch_valid_items, batch_candidates = test_dataset.get_next_batch(batch_users)
		batch_total_items = torch.cat([batch_test_items, batch_valid_items, batch_candidates], -1)

		batch_users = batch_users.to(gpu)
		batch_total_items = batch_total_items.to(gpu)

		batch_score_mat = model.forward_multi_items(batch_users, batch_total_items)

		batch_score_mat = to_np(-batch_score_mat)
		batch_total_items = to_np(batch_total_items)

		# for each test user in a mini-batch
		for idx, test_user in enumerate(batch_users):
			
			total_items = batch_total_items[idx]
			score = batch_score_mat[idx]

			result = np.argsort(score).flatten().tolist()
			ranking_list = np.array(total_items)[result]

			for mode in ['test', 'valid']:
				if mode == 'test':
					target_item = total_items[0]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[1]))
				else:
					target_item = total_items[1]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[0]))
				
				for topk in [5, 10, 20]:
					(h, n, m) = LOO_check(ranking_list_tmp, target_item, topk)
					eval_results[mode]['H' + str(topk)].append(h)
					eval_results[mode]['N' + str(topk)].append(n)
					eval_results[mode]['M' + str(topk)].append(m)

		if is_last_batch: break


	# valid, test
	for mode in ['test', 'valid']:
		for topk in [5, 10, 20]:
			eval_results[mode]['H' + str(topk)] = round(np.asarray(eval_results[mode]['H' + str(topk)]).mean(), 4)
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)
			eval_results[mode]['M' + str(topk)] = round(np.asarray(eval_results[mode]['M' + str(topk)]).mean(), 4)	

	return eval_results



def LOO_AE_evaluate(model, gpu, test_dataset):
	"""Leave-one-out evaluation for deep model

	Parameters
	----------
	model : Pytorch Model
	gpu : if available
	test_dataset : Pytorch Dataset

	Returns
	-------
	eval_results : dict
		summarizes the evaluation results
	"""
	metrics = {'H10':[], 'M10':[], 'N10':[], 'H20':[], 'M20':[], 'N20':[], 'H5':[], 'M5':[], 'N5':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	# for each batch
	while True:
		batch_users, batch_users_R, is_last_batch = test_dataset.get_next_batch_users()

		batch_test_items, batch_valid_items, batch_candidates = test_dataset.get_next_batch(batch_users)
		batch_total_items = torch.cat([batch_test_items, batch_valid_items, batch_candidates], -1)

		batch_users = batch_users.to(gpu)
		batch_users_R = batch_users_R.to(gpu)
		batch_score_mat = model(batch_users, batch_users_R)

		batch_score_mat = to_np(-batch_score_mat)
		batch_total_items = to_np(batch_total_items)

		# for each test user in a mini-batch
		for idx, test_user in enumerate(batch_users):
			
			total_items = batch_total_items[idx]
			score = batch_score_mat[idx][total_items]

			result = np.argsort(score).flatten().tolist()
			ranking_list = np.array(total_items)[result]

			for mode in ['test', 'valid']:
				if mode == 'test':
					target_item = total_items[0]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[1]))
				else:
					target_item = total_items[1]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[0]))
				
				for topk in [5, 10, 20]:
					(h, n, m) = LOO_check(ranking_list_tmp, target_item, topk)
					eval_results[mode]['H' + str(topk)].append(h)
					eval_results[mode]['N' + str(topk)].append(n)
					eval_results[mode]['M' + str(topk)].append(m)

		if is_last_batch: break


	# valid, test
	for mode in ['test', 'valid']:
		for topk in [5, 10, 20]:
			eval_results[mode]['H' + str(topk)] = round(np.asarray(eval_results[mode]['H' + str(topk)]).mean(), 4)
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)
			eval_results[mode]['M' + str(topk)] = round(np.asarray(eval_results[mode]['M' + str(topk)]).mean(), 4)	

	return eval_results



def evaluation(model, gpu, eval_dict, epoch, test_dataset):
	"""evluatie a given model

	Parameters
	----------
	model : Pytorch Model
	gpu : if available
	eval_dict : dict
		for control the training process
	epoch : int
		current epoch
	test_dataset : Pytorch Dataset

	Returns
	-------
	is_improved : is the result improved compared to the last best results
	eval_results : summarizes the evaluation results
	toc-tic : elapsed time for evaluation
	"""

	model.eval()
	with torch.no_grad():

		tic = time.time()

		# MLP, NeuMF
		if model.sim_type == 'network':
			eval_results = LOO_Net_evaluate(model, gpu, test_dataset)

		# BPR, CML
		elif (model.sim_type == 'inner product') or (model.sim_type == 'L2 dist'):
			eval_results = LOO_latent_factor_evaluate(model, test_dataset)

		# CDAE
		elif model.sim_type == 'AE':
			eval_results = LOO_AE_evaluate(model, gpu, test_dataset)

		else:
			assert 'Unknown sim_type'	

		toc = time.time()
		is_improved = False

		for topk in [5, 10, 20]:
			if eval_dict['early_stop'] < eval_dict['early_stop_max']:

				if eval_dict[topk]['best_score'] < eval_results['valid']['H' + str(topk)]:
					eval_dict[topk]['best_score'] = eval_results['valid']['H' + str(topk)]
					eval_dict[topk]['best_result'] = eval_results['valid']
					eval_dict[topk]['final_result'] = eval_results['test']

					is_improved = True
					eval_dict['final_epoch'] = epoch

		if not is_improved:
			eval_dict['early_stop'] += 1
		else:
			eval_dict['early_stop'] = 0

		return is_improved, eval_results, toc-tic



