import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import random
import torch.optim as optim
import pickle
import torch.utils.data
from torch.backends import cudnn
from scipy.sparse import csr_matrix
import math
import bottleneck as bn
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from Utils.utils import *
from Utils.models import *
from Utils.cal_method import *

gpu = 0
dataset = 'yahooR3'
train_pair = np.load('data/'+dataset+'/train.npy', allow_pickle=True).astype(int)
train_dic = np.load('data/'+dataset+'/train_dic.npy', allow_pickle=True).item()

val = np.load('data/'+dataset+'/val.npy', allow_pickle=True).astype(int)
val_dic = np.load('data/'+dataset+'/val_dic.npy', allow_pickle=True).item()

trainval_dic = np.load('data/'+dataset+'/trainval_dic.npy', allow_pickle=True).item()
trainval_mat = np.load('data/'+dataset+'/trainval_mat.npy', allow_pickle=True)

test = np.load('data/'+dataset+'/test.npy', allow_pickle=True)
test_dic = np.load('data/'+dataset+'/test_dic.npy', allow_pickle=True).item()
test_cdd = np.load('data/'+dataset+'/test_cdd.npy', allow_pickle=True).item()

num_user = int(max(train_pair[:,0].max(), val[:,0].max(), test[:,0].max())) + 1
num_item = int(max(train_pair[:,1].max(), val[:,1].max(), test[:,1].max())) + 1
print(num_user, num_item)
print(train_pair.shape, val.shape, test.shape)

train_mat = torch.zeros((num_user, num_item)).cuda(gpu)
train_mat[train_pair[:,0], train_pair[:,1]] = 1

cal_dataset = caldset(num_user, num_item, trainval_dic, val, num_neg=57)
cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=2048, shuffle=True)
cal_loader.dataset.negative_sampling()

model = torch.load('model/UBPR_'+dataset, map_location = 'cuda:'+str(gpu))

cal_method = Gaussian(model, 'UBPR', gpu, torch.sum(train_mat, dim=0), -13.5, 11.6, 0.1)
score, label = cal_method.fit_params(cal_loader, mode='unbiased', const=True, verbose=True)
scores, scores_test, labels_test = cal_method.evaluation(test, verbose=True)