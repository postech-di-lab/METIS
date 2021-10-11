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

gpu_number = 3
save = True

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

    train_divide_data = [[], [], []]
    for i in range(len(train_data)):
        items = train_data[i]
        for j in range(len(items)):
            if j < 3: continue
            subitems = items[:j]
            if len(subitems) > num_position:
                target = items[j]
                train_divide_data[0].append(subitems[-num_position:])
                train_divide_data[1].append(float(len(subitems[-num_position:])))
                train_divide_data[2].append(target)    
            else:
                target = items[j]
                train_divide_data[0].append(subitems)
                train_divide_data[1].append(float(len(subitems)))
                train_divide_data[2].append(target)
    train_data = list(zip(train_divide_data[0], train_divide_data[1], train_divide_data[2]))

    val_divide_data = [[], [], []]
    for i in range(len(val_data)):
        items = val_data[i]
        if len(items[:-1]) > num_position:
            val_divide_data[0].append(items[-num_position-1:-1])
            val_divide_data[1].append(float(len(items[-num_position-1:-1])))
            val_divide_data[2].append(items[-1])
        else:
            val_divide_data[0].append(items[:-1])
            val_divide_data[1].append(float(len(items[:-1])))
            val_divide_data[2].append(items[-1])
    val_data = list(zip(val_divide_data[0], val_divide_data[1], val_divide_data[2]))

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

    print("# Training data: " + str(len(train_data)))
    print("# Validation data: " + str(len(val_data)))
    print("# Testing data: " + str(len(test_data)))
    
    model = SASRec(num_items, embed_dim, num_position, num_head, num_layers, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr) # Adam
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)

    best_mAP = 0.0
    best_mAP_i = 0
    best_test_mAP = 0.0
    best_test_mAP_i = 0

    for epoch in range(1, num_epoch+1):
        print("Epoch: " + str(epoch))
        # train for one epoch
        random.shuffle(train_data)

        train_loss = trainForEpoch(train_data, model, optimizer, scheduler, batch_size)

        val_mAP = validate(val_data, model, k)
        print('# Val mAP: ' + str(round(val_mAP, 4)))
        
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_mAP_i = epoch
            test_mAP = validate(test_data, model, k)
            if test_mAP > best_test_mAP:
                best_test_mAP = test_mAP
                best_test_mAP_i = epoch
            if save == True:
                torch.save(model.state_dict(), f'./Saved_models/{model_name}_{epoch}.model')
                print(f'### Save model at ./Saved_models/{model_name}_{epoch}.model')
                
            print('## Test mAP: ' + str(round(test_mAP, 4)))

        if best_mAP_i + patience < epoch:
#             print('## Best test mAP: ' + str(round(test_mAP, 4)), 'Epoch:', str(best_mAP_i))
            print('## Best test mAP: ' + str(round(best_test_mAP, 4)), 'Epoch:', str(best_test_mAP_i))
            exit(1)


def pad(l, limit, p):
    max_len = limit
    l = list(map(lambda x: [p] * (max_len - min(len(x), limit)) + x[:min(len(x), limit)], l))
    
    return l


def trainForEpoch(train_data, model, optimizer, scheduler, batch_size):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    sum_epoch_target_loss = 0
    for i in range(0, len(train_data), batch_size):
        if i + batch_size >= len(train_data):
            train_batch = train_data[i:]
        else:
            train_batch = train_data[i: i + batch_size]

        sess_, length_, target_ = zip(*train_batch)

        sess = torch.tensor(pad(sess_, int(max(length_)), 0)).to(device) # batch_size * max_length
        target = torch.tensor(target_).to(device)
        length = torch.tensor(length_).to(device) # batch_size
        
        optimizer.zero_grad()

        output = model(sess)
        output[range(len(output)), sess.t()] = float('-inf')
        loss = criterion(output, target)

        loss.backward()
        optimizer.step() 
        scheduler.step()
        
        loss_target = loss.item()
        
        sum_epoch_target_loss += loss_target

        if (i / batch_size) % (len(train_data)/batch_size / 5) == (len(train_data)/batch_size / 5) - 1:
            print('')
            print('[TRAIN] target_loss: %.4f (avg %.4f)' % (loss_target, sum_epoch_target_loss / (i/batch_size + 1)))

    return sum_epoch_target_loss

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
