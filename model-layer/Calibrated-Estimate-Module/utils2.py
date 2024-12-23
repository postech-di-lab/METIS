# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import torch
import matplotlib.pyplot as plt


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y

def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    y[y< thres] = 0
    y[y>=thres] = 1
    return y

def shuffle(x, y):
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]

def ndcg_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    return result_map

def recall_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)
        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()
            recall = np.sum(y_u[pred_top_k]) / max(1, sum(y_u))
            result_map["recall_{}".format(top_k)].append(recall)

    return result_map

def precision_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)
        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()
            recall = np.sum(y_u[pred_top_k]) / top_k
            result_map["precision_{}".format(top_k)].append(recall)

    return result_map

def ECELoss(scores, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.mul(torch.ones_like(scores), 0.5)).type(torch.IntTensor)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.501: # for binary classification
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.cpu().data

def RelDiagram(scores, labels, title=None, n_bins = 10, gpu=0, save=False):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.ones_like(scores) * torch.FloatTensor([0.5])).type(torch.IntTensor)
    accuracies = predictions.eq(labels)

    print(predictions.shape)
    print(accuracies.shape)
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
    print(acc)
    print(conf)
    acc_conf = np.column_stack([acc,conf])
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1/n_bins
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    fig = plt.figure(figsize=(4,4))
    ax = fig.subplots()

    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=1)
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