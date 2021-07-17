import argparse

from Models.BPR import BPR
from Utils.dataset import implicit_CF_dataset_URRD, implicit_CF_dataset_test, implicit_CF_dataset_IR_reg
from Utils.data_utils import read_LOO_settings, load_pickle

import torch
import torch.utils.data as data
import torch.optim as optim

from run import LOO_IR_RRD_run

from pdb import set_trace as bp

def run():

	# gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# dataset setting
	data_path, dataset, LOO_seed = opt.data_path, opt.dataset, opt.LOO_seed

	# for train
	model, lr, batch_size, num_ns = opt.model, opt.lr, opt.batch_size, opt.num_ns
	reg = opt.reg

	# for URRD
	user_topk_dict = load_pickle('for_KD/', 'citeULike.bpr.teacher_topk_dict_0')
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(data_path, dataset, LOO_seed)
	print("User::", user_count, "Item::", item_count, "Interactions::", len(train_interactions))

	# for IR reg
	item_topk_dict = load_pickle('for_KD/', 'citeULike.bpr.teacher_item_topk_dict_0')

	train_dataset = implicit_CF_dataset_URRD(user_count, item_count, train_mat, train_interactions, num_ns, gpu, user_topk_dict, opt.U_T, opt.U_K, opt.U_L)
	IR_reg_train_dataset = implicit_CF_dataset_IR_reg(user_count, item_count, train_mat, train_interactions, num_ns, gpu, item_topk_dict, opt.I_T, opt.I_K, opt.I_L)

	test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)
	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	
	# model
	teacher_dims = 200
	dim = int(teacher_dims * opt.percent)
	model = BPR(user_count, item_count, dim, gpu)

	# optimizer
	model = model.to(gpu)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

	# start train
	LOO_IR_RRD_run(opt, model, gpu, optimizer, train_loader, test_dataset, IR_reg_train_dataset, model_save_path=None)


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

	# for RRD
	parser.add_argument('--model', type=str, default='BPR')
	parser.add_argument('--percent', type=float, default=0.1, help='for student model size')
	parser.add_argument('--URRD_lmbda', type=float, default=0.001)
	parser.add_argument('--U_K', type=int, default=40)
	parser.add_argument('--U_L', type=int, default=40)
	parser.add_argument('--U_T', type=int, default=10)

	# for IR reg
	parser.add_argument('--IR_reg_lmbda', type=float, default=0.0005)
	parser.add_argument('--I_K', type=int, default=20)
	parser.add_argument('--I_L', type=int, default=20)
	parser.add_argument('--I_T', type=int, default=10)	

	opt = parser.parse_args()
	print(opt)

	run()