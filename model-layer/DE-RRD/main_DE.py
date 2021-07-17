import argparse
import os

from Models.BPR import BPR
from Models.DE import BPR_DE

from Utils.dataset import implicit_CF_dataset, implicit_CF_dataset_test
from Utils.data_utils import read_LOO_settings

import torch
import torch.utils.data as data
import torch.optim as optim

from run import LOO_DE_run


def run():

	# gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# for train
	lr, batch_size, num_ns = opt.lr, opt.batch_size, opt.num_ns
	reg = opt.reg

	# dataset
	data_path, dataset, LOO_seed = opt.data_path, opt.dataset, opt.LOO_seed
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(data_path, dataset, LOO_seed)

	train_dataset = implicit_CF_dataset(user_count, item_count, train_mat, train_interactions, num_ns)
	test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)

	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	# Read teacher
	teacher_dims = 200
	teacher_model = BPR(user_count, item_count, teacher_dims, gpu)
		
	with torch.no_grad():
		teacher_model_path = opt.teacher_path + dataset +"/" + 'bpr_0.001_200_0.001.model_0'
		teacher_model = teacher_model.to(gpu)
		teacher_model.load_state_dict(torch.load(teacher_model_path))		
		teacher_user_emb, teacher_item_emb = teacher_model.get_embedding()
		del teacher_model

	# Student model
	dim = int(teacher_dims * opt.percent)
	model = BPR_DE(user_count, item_count, teacher_user_emb, teacher_item_emb, gpu=gpu, student_dim=dim, num_experts=opt.num_expert)
		 
	# optimizer
	model = model.to(gpu)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

	# start train
	LOO_DE_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# training
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--reg', type=float, default=0.001)
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--num_ns', type=int, default=1)

	parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
	parser.add_argument('--max_epoch', type=int, default=1000)
	parser.add_argument('--early_stop', type=int, default=20)
	parser.add_argument('--es_epoch', type=int, default=0)

	# dataset
	parser.add_argument('--data_path', type=str, default='Dataset/')
	parser.add_argument('--dataset', type=str, default='citeULike')
	parser.add_argument('--LOO_seed', type=int, default=0, help='0, 1, 2, 3, 4')

	# for DE
	parser.add_argument('--teacher_path', type=str, default='Saved_models/')
	parser.add_argument('--lmbda_DE', type=float, default=0.01, help='for lmbda_DE')

	parser.add_argument('--end_T', type=float, default=1e-10, help='for MTD_lmbda')
	parser.add_argument('--anneal_size', type=int, default=1e+10, help='T annealing')

	parser.add_argument('--num_expert', type=int, default=30, help='for decoder')
	parser.add_argument('--percent', type=float, default=0.1, help='for student model size')
	
	opt = parser.parse_args()
	print(opt)

	run()

