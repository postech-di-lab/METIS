#!/usr/bin/env python37
# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torch.backends import cudnn

from model import *

import pandas as pd

from dataset import movielens_preprocess

from sklearn.metrics import average_precision_score

#################################################################################

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

#################################################################################

gpu_number = 0
epoch = 30

batch_size = 256
val_batch_size = 512
embed_dim = 128

lr = 0.002

num_epoch = 500
dropout = 0.2 # 0
num_position = 50

num_head = 1 # 1
num_layers = 2 # 2

k = [5, 10, 20]
patience = 10

model_name = 'sasrec'  

##################################################################################

device = torch.device("cuda:" + str(gpu_number)) if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(device) # change allocation of current GPU
print(f"training device: {device, torch.cuda.get_device_name()}")

print("batch_size     : " + str(batch_size))
print("val_batch_size : " + str(val_batch_size))
print("embed_dim      : " + str(embed_dim))
print("lr             : " + str(lr))
print("num_epoch      : " + str(num_epoch))
print("dropout        : " + str(dropout))
print("num_head       : " + str(num_head))
print("num_layers     : " + str(num_layers))

torch.set_printoptions(edgeitems=10)
torch.set_printoptions(linewidth=100)

def main():
    print('Loading data...')

    train_data, val_data, test_data = movielens_preprocess()
    num_items = 3953
    
    test_divide_data = [[], [], []]
    for i in range(len(test_data)):
        items = test_data[i]
        if len(items[:-1]) > num_position:
            test_divide_data[0].append(items[-num_position-1:-1])
            test_divide_data[1].append(float(len(items[-num_position-1:-1])))
            test_divide_data[2].append(items[-1])
        else:
            test_divide_data[0].append(items[:-1])
            test_divide_data[1].append(float(len(items[:-1])))
            test_divide_data[2].append(items[-1])
    test_data = list(zip(test_divide_data[0], test_divide_data[1], test_divide_data[2]))

    print("# Testing data: " + str(len(test_data)))
    
    model = SASRec(num_items, embed_dim, num_position, num_head, num_layers, dropout)
    model.load_state_dict(torch.load(f'./Saved_models/{model_name}_{epoch}.model'))
    model.to(device)

    test_mAP = validate(test_data, model, k)
    print('## Test mAP: ' + str(round(test_mAP, 4)))
    
def pad(l, limit, p):
    max_len = limit
    l = list(map(lambda x: [p] * (max_len - min(len(x), limit)) + x[:min(len(x), limit)], l))
    
    return l

def validate(val_data, model, k):
    model.eval()

    len_val = 0
    mAP = 0.0
    with torch.no_grad():
        for i in range(0, len(val_data), val_batch_size):
            if i + val_batch_size >= len(val_data):
                val_batch = val_data[i:]
            else:
                val_batch = val_data[i: i + val_batch_size]

            val_batch.sort(key=lambda x: x[1], reverse=True)

            sess_, length, target = zip(*val_batch)

            last_item = torch.tensor(list(map(lambda x: x[-1], sess_))).to(device)
            sess = torch.tensor(pad(sess_, int(max(length)), 0)).to(device)
            target = torch.tensor(target).to(device)
            length = torch.tensor(length).to(device)

            output = model(sess)
            
            output[range(len(output)), sess.t()] = float('-inf')
            logits = F.softmax(output, dim = 1)
            
            len_val += sess.shape[0]

            target_onehot = torch.zeros_like(logits)
            target_onehot[range(len(target_onehot)), target] = 1
            mAP += average_precision_score(target_onehot.cpu().detach().numpy(), logits.cpu().detach().numpy(), average='samples') * sess.shape[0]

    return mAP / len_val

if __name__ == '__main__':
    main()
