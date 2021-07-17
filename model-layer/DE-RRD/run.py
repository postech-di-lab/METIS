import time
from copy import deepcopy

import torch
import torch.optim as optim

from Utils.evaluation import evaluation, LOO_print_result, print_final_result
from Utils.loss import relaxed_ranking_loss
from Utils.data_utils import T_annealing


def LOO_IR_RRD_run(opt, model, gpu, optimizer, train_loader, test_dataset, IR_reg_train_dataset, model_save_path=None):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	save= True

	train_loss_arr = []

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	for epoch in range(max_epoch):
		
		tic1 = time.time()
		train_loader.dataset.negative_sampling()
		train_loader.dataset.URRD_sampling()
		IR_reg_train_dataset.IR_reg_sampling()
		epoch_loss = []
		
		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			# Forward Pass
			model.train()

			# Base Loss
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)

			# URRD Loss
			batch_user = batch_user.unique()
			interesting_items, uninteresting_items = train_loader.dataset.get_samples(batch_user)
			interesting_items = interesting_items.to(gpu).type(torch.cuda.LongTensor)
			uninteresting_items = uninteresting_items.to(gpu).type(torch.cuda.LongTensor)

			interesting_prediction = model.forward_multi_items(batch_user, interesting_items)
			uninteresting_prediction = model.forward_multi_items(batch_user, uninteresting_items)

			URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

			# IR regularizer
			batch_item = torch.cat([interesting_items.view((-1,)), uninteresting_items.view((-1,))]).unique()
			interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(batch_item)	

			interesting_users = interesting_users.to(gpu).type(torch.cuda.LongTensor)
			uninteresting_users = uninteresting_users.to(gpu).type(torch.cuda.LongTensor)

			interesting_user_prediction = model.forward_multi_users(interesting_users, batch_item)
			uninteresting_user_prediction = model.forward_multi_users(uninteresting_users, batch_item)

			IR_reg = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)

			# batch loss
			batch_loss = base_loss + opt.URRD_lmbda * URRD_loss + opt.IR_reg_lmbda * IR_reg
			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		train_loss_arr.append(epoch_loss)

		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)



def LOO_URRD_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	save= True

	train_loss_arr = []

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	for epoch in range(max_epoch):
		
		tic1 = time.time()
		train_loader.dataset.negative_sampling()
		train_loader.dataset.URRD_sampling()
		epoch_loss = []
		
		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			# Forward Pass
			model.train()

			# Base Loss
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)

			# URRD Loss
			batch_user = batch_user.unique()
			interesting_items, uninteresting_items = train_loader.dataset.get_samples(batch_user)
			interesting_items = interesting_items.to(gpu).type(torch.cuda.LongTensor)
			uninteresting_items = uninteresting_items.to(gpu).type(torch.cuda.LongTensor)

			interesting_prediction = model.forward_multi_items(batch_user, interesting_items)
			uninteresting_prediction = model.forward_multi_items(batch_user, uninteresting_items)

			URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

			# batch loss
			batch_loss = base_loss + opt.URRD_lmbda * URRD_loss
			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		train_loss_arr.append(epoch_loss)

		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)



def LOO_DE_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	
		save= True

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	current_T = opt.end_T * opt.anneal_size

	# begin training
	for epoch in range(max_epoch):
		
		tic1 = time.time()
		train_loader.dataset.negative_sampling()
		epoch_loss = []

		model.T = current_T
		
		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			# Forward Pass
			model.train()
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)

			DE_loss_user = model.get_DE_loss(batch_user.unique(), is_user=True)
			DE_loss_pos = model.get_DE_loss(batch_pos_item.unique(), is_user=False)
			DE_loss_neg = model.get_DE_loss(batch_neg_item.unique(), is_user=False)

			DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
			batch_loss = base_loss + DE_loss * opt.lmbda_DE

			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

		# annealing
		current_T = T_annealing(epoch, max_epoch, opt.end_T * opt.anneal_size, opt.end_T)
		if current_T < opt.end_T:
			current_T = opt.end_T

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)



def LOO_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	
		save= True

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	# begin training
	for epoch in range(max_epoch):
		
		tic1 = time.time()
		train_loader.dataset.negative_sampling()
		epoch_loss = []
		
		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			# Forward Pass
			model.train()
			output = model(batch_user, batch_pos_item, batch_neg_item)
			batch_loss = model.get_loss(output)
			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)
