import os
import sys
import csv
import pdb
import copy
import random
import itertools
import numpy as np
import scipy.sparse as sp

from time import time
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix

def build_hist(uirt): # NOTE: this UIRT extracted from only the training data
    it_dict = {}
    uit_dict = {}
    iut_dict = {}
    userset, itemset = set(), set()            
    for u, i, _, t in uirt:
        if u not in uit_dict: uit_dict[u] = {'item':[], 'time':[]}
        uit_dict[u]['item'].append(i)
        uit_dict[u]['time'].append(float(t)) 

        if i not in iut_dict: iut_dict[i] = {'user':[], 'time':[]}
        iut_dict[i]['user'].append(u)
        iut_dict[i]['time'].append(float(t)) 
        
        if i not in it_dict: it_dict[i] = {'time':[]}
        it_dict[i]['time'].append(float(t))

        userset.add(u)
        itemset.add(i)

    # Sorting for user-item-time dictioanry
    userhist = {}            
    for u in uit_dict:
        items = np.array(uit_dict[u]['item'])
        times = np.array(uit_dict[u]['time'])

        # sort them chronologically
        sorted_idx = times.argsort()
        sorted_items = items[sorted_idx]
        sorted_times = times[sorted_idx]                
        
        userhist[u] = [sorted_items, sorted_times]

    # Sorting for item-user-time dictioanry
    neighborhist = {}            
    for i in iut_dict:
        users = np.array(iut_dict[i]['user'])
        times = np.array(iut_dict[i]['time'])

        # sort them chronologically
        sorted_idx = times.argsort()
        sorted_users = users[sorted_idx]
        sorted_times = times[sorted_idx]                
        
        neighborhist[i] = [sorted_users, sorted_times]

    # Sorting for item-time dictioanry    
    itemhist = {}
    for i in it_dict:
        times = np.array(it_dict[i]['time'])

        # sort them chronologically
        sorted_idx = times.argsort()        
        sorted_times = times[sorted_idx]                
        
        itemhist[i] = sorted_times        
        
    return userhist, itemhist, neighborhist

def build_hist_neighbor(uirt, order): 

    nuit_dict = {}
    for u, i, _, t in uirt:
        if u not in nuit_dict: unit_dict[u] = {}
        if i not in nuit_dict: unit_dict[u][i] = []

        consumed_times = find_consumed_times(u, i, order)

        unit_dict[u][i] = consumed_times



    
    pdb.set_trace()



def replace_id2idx(trn, vld, tst):
    
    def build_dict(category):
        category = list(set(category))

        cate_dict = {}
        for i, c in enumerate(category): cate_dict[c] = i + 1
        return cate_dict

    def id2idx(uir, udict, idict): # Convert IDs in string into IDs in numbers
        newuir = []
        for i in range(len(uir)):
            user, item, rating, time = uir[i] 
            newuir.append([udict[user], idict[item], rating, time])
        return newuir

    trn_users = [i[0] for i in trn] 
    trn_items = [i[1] for i in trn] 
    
    user_dict = build_dict(trn_users)
    item_dict = build_dict(trn_items)
    
    trn = id2idx(trn, user_dict, item_dict)
    vld = id2idx(vld, user_dict, item_dict)
    tst = id2idx(tst, user_dict, item_dict)
    
    return trn, vld, tst, user_dict, item_dict

def load_raw_data(fn):
    print('Load ' + fn)
    rawdata = [l for l in csv.reader(open(fn))]
    return rawdata

def find_negatives(dataset):
    NUMNEG = 100 # The number of negative items for evaluation
    
    trn, vld, tst = dataset
    
    allitems = set([i[1] for i in trn])
    
    uidict = {} # {u: [items consumed by user u]}
    for i in range(len(trn)):
        user, item, _, _ = trn[i] # user ID, item ID, rating, time
        if user not in uidict: uidict[user] = []
        uidict[user].append(item)
    
    for i in range(len(vld)):
        user, item, _, _ = vld[i]
            
        useritems = set(uidict[user] + [item]) # Target item and a user's consumed items
        negative_items = random.sample(list(allitems - useritems), NUMNEG)
        
        vld[i] = vld[i][:-2] + negative_items # Append negative items for evaluation
    
    # Do the same thing like the validation
    for i in range(len(tst)):
        user, item, _, _ = tst[i]
        
        useritems = set(uidict[user] + [item])
        negative_items = random.sample(list(allitems - useritems), NUMNEG) 
        
        tst[i] = tst[i][:-2] + negative_items
    
    return trn, vld, tst
    

dn = sys.argv[1] + '/' if not sys.argv[1].endswith('/') else sys.argv[1]
data_path = dn+'split/' if 'split/' not in dn else dn

print('\n🧰 Building a dataset for training the recommender system \n')

for fn in os.listdir(data_path):
    if 'trn' in fn: trndata_name = data_path+fn
    if 'vld' in fn: vlddata_name = data_path+fn
    if 'tst' in fn: tstdata_name = data_path+fn

# Load datasets and review features from csv format
trndata = load_raw_data(trndata_name)
vlddata = load_raw_data(vlddata_name)
tstdata = load_raw_data(tstdata_name)

trndata, org_vlddata, org_tstdata, user2id_dict, item2id_dict = replace_id2idx(trndata, vlddata, tstdata)

trndata, vlddata, tstdata = find_negatives([trndata, copy.deepcopy(org_vlddata), copy.deepcopy(org_tstdata)])

print('\nTRN:{}\tVLD:{}\tTST:{}'.format(len(trndata), len(vlddata), len(tstdata)))

user_index = [i[0] for i in trndata] 
item_index = [i[1] for i in trndata] 
n_users = max(user_index)+1
n_items = max(item_index)+1

# # Compute the neighbor relation 
# print("\nGenerating adjacency matrix")
# s = time()
# intM = csr_matrix((np.ones(len(user_index)), (user_index, item_index)),
#                                shape=(n_users, n_items)) 
# adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
# adj_mat = adj_mat.tolil()
# R = intM.tolil()
# adj_mat[:n_users, n_users:] = R
# adj_mat[n_users:, :n_users] = R.T
# # adj_mat = adj_mat.todok()
# end = time()
# print("costing {:.2}s for computing the adjacency matrix".format(end-s))

print('\nBuilding user-SeqHist dictionary')
userhist, itemhist, neighborhist = build_hist(trndata)
userhist_wvld, itemhist_wvld, neighborhist_wvld = build_hist(trndata+org_vlddata)

print('\n📂 Starting to save datasets')
dataset_name = dn.split('/')[0] 
base_path = dataset_name+'/rec/'
if not os.path.exists(base_path): os.makedirs(base_path)

data_path = base_path

np.save(open(data_path+'trn','wb'), np.array(trndata).astype(float).astype(int))
np.save(open(data_path+'vld','wb'), np.array(vlddata).astype(float).astype(int))
np.save(open(data_path+'tst','wb'), np.array(tstdata).astype(float).astype(int))
# Below dictionaries are for recovering the origial user and item names
np.save(open(data_path+'user_dict','wb'), user2id_dict)
np.save(open(data_path+'item_dict','wb'), item2id_dict)
# Save neighborhood info.
# sp.save_npz(data_path + '/1order.npz', adj_mat.tocsr())
# Save user-SeqHist 
np.save(open(data_path+'userhist','wb'), userhist)
np.save(open(data_path+'userhist_wvld','wb'), userhist_wvld)

np.save(open(data_path+'itemhist','wb'), itemhist)
np.save(open(data_path+'itemhist_wvld','wb'), itemhist_wvld)

np.save(open(data_path+'neighborhist','wb'), neighborhist)
np.save(open(data_path+'neighborhist_wvld','wb'), neighborhist_wvld)

np.save(open(data_path+'vld_info','wb'), np.array(org_vlddata))
np.save(open(data_path+'tst_info','wb'), np.array(org_tstdata))

print('\nDatasets saved to the data directory: {}\n'.format(data_path))
