from utils import train_dataset, test_dataset, evaluate, print_result, read_settings
from copy import deepcopy
import time

import torch
import torch.utils.data as data
import torch.optim as optim

import random
import numpy as np

from IPython.display import display, clear_output


def T_annealing(epoch, max_epoch, initial_T, end_T):
	new_T = initial_T * ((end_T / initial_T) ** (epoch / max_epoch))
	return new_T


def run_base(run_dict, model):

	optimizer = optim.Adam(model.parameters(), lr=run_dict['lr'], weight_decay=run_dict['reg'])

	train_loader, test_dataset = run_dict['train_loader'], run_dict['test_dataset']
	gpu = run_dict['gpu']
	early_stop = 0.

	history = []
	# Begin training
	for epoch in range(run_dict['max_epoch']+1):
		
		train_loader.dataset.negative_sampling()
		epoch_loss = []

		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			model.train()

			## Base Model
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)
			batch_loss = base_loss
			epoch_loss.append(batch_loss)
			
			## Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		
		# Evaluation
		if epoch % run_dict['eval_period'] == 0:
			
			model.eval()
			with torch.no_grad():
				eval_results = evaluate(model, gpu, train_loader, test_dataset)
				current_R50 = eval_results['test']['R50']

			history.append(current_R50)
			clear_output()
			display('Epoch [%d/%d], Recall@50: %.4f' % (epoch, run_dict['max_epoch'], current_R50))

	clear_output()
	print("Train Done!, Recall@50: %.4f" % (current_R50))

	return history


def run_FTD(run_dict, model):

	optimizer = optim.Adam(model.parameters(), lr=run_dict['lr'], weight_decay=run_dict['reg'])

	train_loader, test_dataset = run_dict['train_loader'], run_dict['test_dataset']
	gpu = run_dict['gpu']
	early_stop = 0.

	history = []
	# Begin training
	for epoch in range(run_dict['max_epoch']+1):
		
		train_loader.dataset.negative_sampling()
		epoch_loss = []

		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			model.train()

			## Base Model
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)

			## Topology Distillation
			TD_loss  = model.get_TD_loss(batch_user.unique(), torch.cat([batch_pos_item, batch_neg_item], 0).unique())
			batch_loss = base_loss + TD_loss * run_dict['lmbda_TD']
			epoch_loss.append(batch_loss)
			
			## Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		
		# Evaluation
		if epoch % run_dict['eval_period'] == 0:
			
			model.eval()
			with torch.no_grad():
				eval_results = evaluate(model, gpu, train_loader, test_dataset)
				current_R50 = eval_results['test']['R50']

			history.append(current_R50)
			clear_output()
			display('Epoch [%d/%d], Recall@50: %.4f' % (epoch, run_dict['max_epoch'], current_R50))

	clear_output()
	print("Train Done!, Recall@50: %.4f" % (current_R50))

	return history


def run_HTD(run_dict, model):

	optimizer = optim.Adam(model.parameters(), lr=run_dict['lr'], weight_decay=run_dict['reg'])

	train_loader, test_dataset = run_dict['train_loader'], run_dict['test_dataset']
	gpu = run_dict['gpu']
	early_stop = 0.
	current_T = 1.

	history = []
	# Begin training
	for epoch in range(run_dict['max_epoch']+1):
		
		train_loader.dataset.negative_sampling()
		epoch_loss = []
		model.T = current_T

		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			model.train()

			## Base Model
			output = model(batch_user, batch_pos_item, batch_neg_item)
			base_loss = model.get_loss(output)

			## Group Assignment
			GA_loss_user = model.get_GA_loss(batch_user.unique(), is_user=True)
			GA_loss_item = model.get_GA_loss(torch.cat([batch_pos_item, batch_neg_item], 0).unique(), is_user=False)
			GA_loss = GA_loss_user + GA_loss_item

			## Topology Distillation
			TD_loss  = model.get_TD_loss(batch_user.unique(), torch.cat([batch_pos_item, batch_neg_item], 0).unique())
			HTD_loss = TD_loss * run_dict['alpha'] + GA_loss * (1 - run_dict['alpha'])
			batch_loss = base_loss + HTD_loss * run_dict['lmbda_TD']
			epoch_loss.append(batch_loss)

			## Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		
		# Evaluation
		if epoch % run_dict['eval_period'] == 0:
			
			model.eval()
			with torch.no_grad():
				eval_results = evaluate(model, gpu, train_loader, test_dataset)
				current_R50 = eval_results['test']['R50']
				
			history.append(current_R50)
			clear_output()
			display('Epoch [%d/%d], Recall@50: %.4f' % (epoch, run_dict['max_epoch'], current_R50))

		# T annealing (optional)
		current_T = T_annealing(epoch, run_dict['max_epoch'], 1, 1e-10)
		if current_T < 1e-10:
			current_T = 1e-10

	clear_output()
	print("Train Done!, Recall@50: %.4f" % (current_R50))

	return history
