import argparse

from Models.BPR import BPR

from Utils.dataset import implicit_CF_dataset, implicit_CF_dataset_test
from Utils.data_utils import read_LOO_settings

import torch
import torch.utils.data as data
import torch.optim as optim

from run import LOO_run

def run():

	# gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# for training
	model, lr, batch_size, num_ns = opt.model, opt.lr, opt.batch_size, opt.num_ns
	reg = opt.reg
	save = opt.save

	# dataset
	data_path, dataset, LOO_seed = opt.data_path, opt.dataset, opt.LOO_seed
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(data_path, dataset, LOO_seed)

	train_dataset = implicit_CF_dataset(user_count, item_count, train_mat, train_interactions, num_ns)
	test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)

	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	dim = opt.dim
	model = BPR(user_count, item_count, dim, gpu)

	# optimizer
	model = model.to(gpu)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
	
	print("User::", user_count, "Item::", item_count, "Interactions::", len(train_interactions))

	# to save model
	model_save_path = 'None'
	if (save == 1):
		model_save_path = './Saved_models/' + opt.dataset +"/" + str(opt.model) +"_" + str(opt.lr) + "_" + str(opt.dim) + "_" + str(opt.reg) +'.model' + "_" + str(opt.LOO_seed)
	
	# start train
	LOO_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# model
	parser.add_argument('--model', type=str, default='BPR')
	parser.add_argument('--dim', type=int, default=20, help='embedding dimensions')

	# training
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--reg', type=float, default=0.001, help='for L2 regularization')
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--num_ns', type=int, default=1, help='number of negative samples')

	parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')

	parser.add_argument('--max_epoch', type=int, default=1000)
	parser.add_argument('--early_stop', type=int, default=20)
	parser.add_argument('--es_epoch', type=int, default=0, help='evaluation start epoch')
	parser.add_argument('--save', type=int, default=0, help='0: false, 1:true')

	# dataset
	parser.add_argument('--data_path', type=str, default='Dataset/')
	parser.add_argument('--dataset', type=str, default='citeULike', help='citeULike, Foursquare')
	parser.add_argument('--LOO_seed', type=int, default=0, help='0')

	opt = parser.parse_args()
	print(opt)

	run()