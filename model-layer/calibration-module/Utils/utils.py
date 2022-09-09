import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt

def is_visited(base_dict, user_id, item_id):
    if user_id in base_dict and item_id in base_dict[user_id]:
        return True
    else:
        return False

def get_user_item_count_dict(interactions):
	user_count_dict = {}
	item_count_dict = {}

	for user, item in interactions:
		if user not in user_count_dict:
			user_count_dict[user] = 1
		else:
			user_count_dict[user] += 1

		if item not in item_count_dict:
			item_count_dict[item] = 1
		else:
			item_count_dict[item] += 1

	return user_count_dict, item_count_dict

def get_adj_mat(user_count, item_count, train_interactions):
	user_count_dict, item_count_dict = get_user_item_count_dict(train_interactions)

	A_indices, A_values = [[], []], []
	A_T_indices, A_T_values = [[], []], []
	for user, item in train_interactions:
		A_indices[0].append(user)
		A_indices[1].append(item)
		A_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

		A_T_indices[0].append(item)
		A_T_indices[1].append(user)
		A_T_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

	A_indices = torch.LongTensor(A_indices)
	A_values = torch.FloatTensor(A_values)

	A = torch.sparse.FloatTensor(A_indices, A_values, torch.Size([user_count, item_count]))

	A_T_indices = torch.LongTensor(A_T_indices)
	A_T_values = torch.FloatTensor(A_T_values)

	A_T = torch.sparse.FloatTensor(A_T_indices, A_T_values, torch.Size([item_count, user_count]))

	return A, A_T

class traindset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_item, train_dic, train_pair, num_neg, gpu):
        super(traindset, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.train_dic = train_dic
        self.train_pair = train_pair
        self.gpu = gpu
        
    def negative_sampling(self):      
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.num_item)), size = 5 * len(self.train_pair) * self.num_neg)
        
        sample_idx = 0
        for user, pos_item in self.train_pair:
            ns_count = 0
            while True:
                neg_item = sample_list[sample_idx]
                sample_idx += 1
                if not is_visited(self.train_dic, user, neg_item):
                    self.train_arr.append((user, pos_item, neg_item))
                    ns_count += 1
                    if ns_count == self.num_neg:
                        break
        
        #self.train_arr = torch.LongTensor(np.asarray(self.train_arr)).cuda(self.gpu)

    def __len__(self):
        return len(self.train_pair) * self.num_neg
        
    def __getitem__(self, idx):
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]

class caldset(torch.utils.data.Dataset):

    def __init__(self, num_user, num_item, train_dic, train_pair, num_neg):
        super(caldset, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.train_dic = train_dic
        self.train_pair = train_pair
        
    def negative_sampling(self):      
        self.train_ui = []
        self.train_label = []
        sample_list = np.random.choice(list(range(self.num_item)), size = 10 * len(self.train_pair) * self.num_neg)
        
        sample_idx = 0
        for user, pos_item in self.train_pair:
            self.train_ui.append((user, pos_item))
            self.train_label.append(1)
            ns_count = 0
            while True:
                neg_item = sample_list[sample_idx]
                sample_idx += 1
                if not is_visited(self.train_dic, user, neg_item):
                    self.train_ui.append((user, neg_item))
                    self.train_label.append(0)
                    ns_count += 1
                    if ns_count == self.num_neg:
                        break

    def __len__(self):
        return len(self.train_pair) * (self.num_neg+1)
        
    def __getitem__(self, idx):
        return self.train_ui[idx][0], self.train_ui[idx][1], self.train_label[idx]
    
    ''

def evaluate_val(K1, K2, K3, score_matrix, test_dic):
    score_matrix = score_matrix.numpy()
    num_user = score_matrix.shape[0]
    num_item = score_matrix.shape[1]

    NDCG = []
    Recall = []
    for u_test in range(num_user):
        if u_test in test_dic:
            item_pos = np.array(test_dic[u_test])
            item_cdd = np.array(item_pos.tolist() + np.random.randint(num_item, size=100).tolist())

            score_cdd = score_matrix[u_test][item_cdd]
            rank_cdd = np.argsort(-score_cdd)
            
            rank_test = []
            for i_test in item_pos:
                test_idx = np.where(item_cdd == i_test)[0][0]
                rank_test.append(np.where(rank_cdd == test_idx)[0][0])
            rank_test = np.asarray(rank_test)
            if len(rank_test) < 1:
                print(u_test)
            
            idcg1 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K1))])
            idcg2 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K2))])
            idcg3 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K3))])
            dcg = 1 / np.log2(rank_test + 2)
            NDCG.append((np.sum(dcg*(rank_test<K1)) / idcg1, np.sum(dcg*(rank_test<K2)) / idcg2, np.sum(dcg*(rank_test<K3)) / idcg3))
            
            Recall.append(((rank_test < K1).mean(), (rank_test < K2).mean(), (rank_test < K3).mean()))

    np.set_printoptions(precision=4)
    print(np.mean(np.asarray(NDCG), axis=0, keepdims=True), np.mean(np.asarray(Recall), axis=0, keepdims=True))

def evaluate_test(K1, K2, K3, score_matrix, test_dic, test_cdd, test_mode):
    score_matrix = score_matrix.numpy()
    num_user = score_matrix.shape[0]
    num_item = score_matrix.shape[1]

    NDCG = []
    Recall = []
    for u_test in range(num_user):
        if u_test in test_dic:
            item_pos = np.array(test_dic[u_test])
            if test_mode == 'cdd':
                item_cdd = np.array(test_cdd[u_test])
            elif test_mode == 'ran':
                item_cdd = np.array(item_pos.tolist() + np.random.randint(num_item, size=100).tolist())
            elif test_mode == 'full':
                item_cdd = np.arange(num_item) #full eval

            score_cdd = score_matrix[u_test][item_cdd]
            rank_cdd = np.argsort(-score_cdd)
            
            rank_test = []
            for i_test in item_pos:
                test_idx = np.where(item_cdd == i_test)[0][0]
                rank_test.append(np.where(rank_cdd == test_idx)[0][0])
            rank_test = np.asarray(rank_test)
            if len(rank_test) < 1:
                print(u_test)

            idcg1 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K1))])
            idcg2 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K2))])
            idcg3 = sum([1/np.log2(l+2) for l in range(min(len(item_pos), K3))])
            dcg = 1 / np.log2(rank_test + 2)
            NDCG.append((np.sum(dcg*(rank_test<K1)) / idcg1, np.sum(dcg*(rank_test<K2)) / idcg2, np.sum(dcg*(rank_test<K3)) / idcg3))
            
            Recall.append(((rank_test < K1).mean(), (rank_test < K2).mean(), (rank_test < K3).mean()))

    np.set_printoptions(precision=4)
    print(np.mean(np.asarray(NDCG), axis=0, keepdims=True), np.mean(np.asarray(Recall), axis=0, keepdims=True))

def ECELoss(scores, labels, n_bins=15, gpu=0):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.ones_like(scores) * torch.FloatTensor([0.5]).cuda(gpu)).type(torch.IntTensor).cuda(gpu)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1).cuda(gpu)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.51: # for binary classification
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.cpu().data

def MCELoss(scores, labels, n_bins=15, gpu=0):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.ones_like(scores) * 0.5).type(torch.IntTensor).cuda(gpu)
    accuracies = predictions.eq(labels)

    ce = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0001:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.51:
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            ce.append(torch.abs(avg_confidence_in_bin - accuracy_in_bin))

    return torch.FloatTensor(ce).max().cpu().data

def RelDiagram(scores, labels, title='GammaCal', n_bins = 10, gpu=0, save=False):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.ones_like(scores) * torch.FloatTensor([0.5]).cuda(gpu)).type(torch.IntTensor).cuda(gpu)
    accuracies = predictions.eq(labels)

    # calculate accuracy in each bin
    acc = []
    conf = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            acc.append(accuracy_in_bin.cpu().data)

            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.51:
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            conf.append(avg_confidence_in_bin.cpu().data)
        else:
            acc.append(torch.FloatTensor([0]))
            conf.append(torch.FloatTensor([0]))
    
    acc_conf = np.column_stack([acc,conf])
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1/n_bins
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    fig = plt.figure(figsize=(4,4))
    ax = fig.subplots()

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=1)

    # Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "blue", label="Accuracy", zorder = 2)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.legend(handles = [gap_plt, output_plt], prop={'size': 13}, loc=9)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.plot([0.5,1], [0.5,1], linestyle = "--", c = 'gray', lw='3', zorder = 3)
    ax.plot([0,0.5], [1,0.5], linestyle = "--", c = 'gray', lw='3', zorder = 4)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Calibrated Prob.", fontsize=20, color = "black")
    ax.set_ylabel("Accuracy", fontsize=20, color = "black")
    
    if save:
        fig.savefig('figure/RD_'+title+'.png', bbox_inches="tight")

    return acc

def ScoreDist(prob, title='val set', save=False):
    prob_1d = prob.detach().cpu().numpy()
    weight_1d = np.ones_like(prob_1d) / len(prob_1d)
    
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    ax.hist(prob_1d, bins=np.arange(0, 1.01, 0.1), histtype='stepfilled', weights=weight_1d, edgecolor = "black", color = "blue")

    ax.set_ylabel('% of samples', fontsize=20, color = "black")
    ax.set_xlabel('Confidence', fontsize=20, color = "black")

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_tick_params(labelsize=20)

    if save:
        fig.savefig('figure/SD_'+title+'.png', bbox_inches="tight")