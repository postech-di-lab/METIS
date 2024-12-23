# -*- coding: utf-8 -*-

import numpy as np
import os
import pdb
import math


data_dir = "./data"

def load_data(name="coat"):

    if name == "coat":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir, "train.ascii")
        test_file = os.path.join(data_set_dir, "test.ascii")

        with open(train_file, "r") as f:
            x_train = []
            for line in f.readlines():
                x_train.append(line.split())

            x_train = np.array(x_train).astype(int)

        with open(test_file, "r") as f:
            x_test = []
            for line in f.readlines():
                x_test.append(line.split())

            x_test = np.array(x_test).astype(int)

        #print("===>Load from {} data set<===".format(name))
        #print("[train] rating ratio: {:.6f}".format((x_train>0).sum() / (x_train.shape[0] * x_train.shape[1])))
        #print("[test]  rating ratio: {:.6f}".format((x_test>0).sum() / (x_test.shape[0] * x_test.shape[1])))

    elif name == "yahoo":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-test.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)
        #print("===>Load from {} data set<===".format(name))
        #print("[train] num data:", x_train.shape[0])
        #print("[test]  num data:", x_test.shape[0])

        return x_train[:,:-1], x_train[:,-1], \
            x_test[:, :-1], x_test[:,-1]

    elif name == 'kuai':
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir, "user.txt")
        test_file = os.path.join(data_set_dir, "random.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                lst = line.strip().split(',')
                lst[2] = int(float(lst[2]))
                x_train.append(lst)
        x_train = np.array(x_train).astype(int)
        # print(x_train[:3])

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                lst = line.strip().split(',')
                lst[2] = int(float(lst[2]))
                x_test.append(lst)
        x_test = np.array(x_test).astype(int)
        #print("===>Load from {} data set<===".format(name))
        #print("[train] num data:", x_train.shape[0])
        #print("[test]  num data:", x_test.shape[0])

        return x_train[:,:-1], x_train[:,-1], \
            x_test[:, :-1], x_test[:,-1]

    else:
        print("Cant find the data set",name)
        return

    return x_train, x_test


def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y

def trainval_split(x_train, y_train, num_user, num_item, start_user):
    u = start_user
    val_idx = []
    val_dic = {}
    train_dic = {}
    idxs = []
    for idx, row in enumerate(x_train):    
        if row[0] == u:
            idxs.append(idx)
        else:
            num_tv = len(idxs)
            num_v = math.ceil(num_tv * 0.1)
            num_t = num_tv - num_v
            idx_t = idxs[:num_t]
            idx_v = idxs[num_t:]

            val_idx += idx_v
            val_dic[row[0]-1] = idx_v
            train_dic[row[0]-1] = idx_t  
            
            u = row[0]
            idxs = [row[1]]
            
    num_tv = len(idxs)
    num_v = math.ceil(num_tv * 0.1)
    num_t = num_tv - num_v
    idx_t = idxs[:num_t]
    idx_v = idxs[num_t:]

    val_idx += idx_v
    val_dic[row[0]-1] = idx_v
    train_dic[row[0]-1] = idx_t  
    
    u = row[0]
    idxs = [row[1]]
    
    # ## check
    # for u in range(1, num_user):
    #     if u in val_dic:
    #         num_val = len(val_dic[u])
    #         if num_val < 1:
    #             print(u, num_val)
                
    # for u in range(1, num_user):
    #     if u in train_dic:
    #         num_train = len(train_dic[u])
    #         if num_train < 1:
    #             print(u, num_train)
    
    ## splitdddd
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    
    train_idx = list(set(np.arange(len(x_train))) - set(val_idx))

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    
    return train_dic, x_train, y_train, val_dic, x_val, y_val