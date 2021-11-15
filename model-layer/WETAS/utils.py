import numpy as np
import torch
from sklearn import metrics

def compute_wacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_dacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_auc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    return metrics.auc(fpr, tpr)

def compute_bestf1(score, label, return_threshold=False):
    if isinstance(score, torch.Tensor):
        score = score.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()

    indices = np.argsort(score)[::-1]
    sorted_score = score[indices]
    sorted_label = label[indices]
    true_indices = np.where(sorted_label == 1)[0]

    bestf1 = 0.0
    best_threshold=None
    T = sum(label)
    for _TP, _P in enumerate(true_indices):
        TP, P = _TP + 1, _P + 1
        precision = TP / P
        recall = TP / T
        f1 = 2 * (precision*recall)/(precision+recall)
        threshold = sorted_score[_P] - np.finfo(float).eps
        if f1 > bestf1: # and threshold <= 0.5:
            bestf1 = f1
            best_threshold = sorted_score[_P] - np.finfo(float).eps
            #best_threshold = (sorted_score[_P-1] + sorted_score[_P]) / 2
    if return_threshold:
        return bestf1, best_threshold
    else:
        return bestf1

def compute_auprc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    return metrics.average_precision_score(label, pred)

def compute_precision_recall(result):
    TP, T, P = result
    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    if T != 0: recall = TP / T
    if P != 0: precision = TP / P

    if recall==0.0 or precision==0.0:
        f1 = 0.0
    else:
        f1 = 2*(recall*precision)/(recall+precision)
    return precision, recall, f1, IoU
