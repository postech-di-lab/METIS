import pdb
import os
import sys
import csv
import json
import random
import numpy as np

from datetime import datetime
from collections import Counter

def drop_item_interaction(data, i_drop_ratio):
    trn, vld, tst = [], [], []

    data.sort(key=lambda x:x[-1]) # sort data by date (in unix timestamp)    

    if i_drop_ratio == 0: return data # no processing

    print('✂️ Item-based data dropping with a ratio of {}'.format(i_drop_ratio))
    idict={}
    for row in data:
        u,i,r,t = row
        if i not in idict: idict[i] = []

        idict[i].append(row)

    output = []
    for i in idict:
        insts = idict[i]

        numinsts = len(insts)
        targetnum = int(numinsts * (1-i_drop_ratio))

        idx = list(range(numinsts))
        random.shuffle(idx)

        target_idx = idx[:targetnum]

        filtered_data = np.stack(insts)[target_idx]

        output.append(filtered_data)
    newdata = np.concatenate(output)
    newdata = newdata[newdata[:,-1].astype(float).argsort()] # sort by the date

    newdata = [[i[0], i[1], float(i[2]), int(i[3])] for i in newdata]

    # # newdata = newdata.tolist()

    # items = [i[1] for i in data]
    # newitems = [i[1] for i in newdata]

    # org_item_inter = len(items)/len(set(items))

    print('\t# Data: from {} to {}\n'.format(len(data), len(newdata)))

    return newdata

    # data = newdata    

def split_by_month(data, rating_only, keepratio, i_drop_ratio): #, isdense, isoverlap, isnew_amazon, i_drop_ratio):
    trn, vld, tst = [], [], []
    
    # Drop data        
    random.shuffle(data)
    keepnum = int(len(data) * keepratio)
    
    print(len(data))
    
    data = data[:keepnum]

    data = drop_item_interaction(data, i_drop_ratio) # drop item interactions
    
    assert type(data[0][-1]) in [int, float]

    data.sort(key=lambda x:x[-1]) # sort data by date (in unix timestamp)        
    
    numdata = len(data)
    
    times = np.array([float(i[-1]) for i in data])
    
    oldest_time = max(times)
    start_test_time = oldest_time - 30 * 24 * 60 * 60 # (30 days)
    start_valid_time = start_test_time - 30 * 24 * 60 * 60 # (30 days)
    
    tst_idx = times >= start_test_time
    vld_idx = (times >= start_valid_time) * (times < start_test_time)
    trn_idx = ~(tst_idx + vld_idx)

    trn_end_idx = np.where(trn_idx == True)[0].max()
    vld_end_idx = np.where(vld_idx == True)[0].max()

    trn = data[:trn_end_idx]
    vld = data[trn_end_idx:vld_end_idx]
    tst = data[vld_end_idx:]    
    

    # User core
    users = np.array(trn)[:,0]
    usercnt = Counter(users)

    filter_trn = []
    for row in trn:
        uid, _, _, _ = row

        cnt = usercnt[uid]

        if cnt < 10: continue

        filter_trn.append(row)   

    print('Filtering TRN: from {} to {}'.format(len(trn), len(filter_trn)))

    newdensity = len(filter_trn)/len(set(np.array(filter_trn)[:,0]))
    print(newdensity)

    trn = filter_trn    
    
    # Filter out new (cold-start) users and items
    trnusers = set([i[0] for i in trn])
    trnitems = set([i[1] for i in trn])

    vld = [row for row in vld if (row[0] in trnusers and row[1] in trnitems)]
    tst = [row for row in tst if (row[0] in trnusers and row[1] in trnitems)]

    print('\nTraining data:\t\t {}'.format(len(trn)))
    print('Validation data:\t {}'.format(len(vld)))
    print('Test data:\t\t {}'.format(len(tst)))
    print('\n# of total data:\t {} / {}'.format(len(trn) + len(vld) + len(tst), len(data)))
                                        
    # if i_drop_ratio >0:
    #     org_item_inter = len(items)/len(set(items))
    #     print('\tAvg inter. per item: from {:.3} to {:.3}'.format(org_item_inter,len([i[1] for i in trn])/len(set(trnitems))))

    return trn, vld, tst
 
fn = sys.argv[1]

keepratio = 1 # keepratio : to reduce the size of large data (e.g., Google review)
i_drop_ratio = 0
if len(sys.argv) == 3:
    # keepratio = float(sys.argv[2])
    i_drop_ratio = float(sys.argv[2])

#   #   #   #   #    #

#  10-core for users #

#   #   #   #   #    #

isdense = False
isoverlap = False
rating_only = False


# if len(sys.argv) == 3:
#     # isoverlap = bool(sys.argv[2])
#     # isdense = False    
#     isdense = bool(sys.argv[2] == 'True')
    
# elif len(sys.argv) == 4:
    
#     isdense = bool(sys.argv[2] == 'True')
    
#     i_drop_ratio = float(sys.argv[3])
    
if not fn.startswith('reviews_'):
    rating_only = True

print('\n'+fn+'\n')
        


if fn == 'ml-100k':
    mydata = [i.split() for i in open('ml-100k/u.data')]
elif fn.startswith('uirt'):
    mydata = np.load(fn, allow_pickle=True).tolist()
    
    for i in range(len(mydata)): mydata[i][-1] = int(mydata[i][-1]) # time must be number to sort
    
elif 'yelp19' in fn:    
    data = [json.loads(l) for l in open(fn)]
    mydata = []    
    for l in data:
        utime = datetime.strptime(l['date'].split()[0], "%Y-%m-%d").strftime('%s')
        mydata.append([l['user_id'], l['business_id'], l['stars'], utime])  
elif rating_only == True:
    mydata = [[i[0], i[1], i[2], int(i[3])] for i in csv.reader(open(fn))]   
else:
    data = [json.loads(l) for l in open(fn)]
    # This code assumes Amazon data JSON format. The Amazon data are avaialble at 'http://jmcauley.ucsd.edu/data/amazon/'.

    mydata = [[l['reviewerID'], l['asin'], l['overall'], int(float(l['unixReviewTime']))] for l in data]

trndata, vlddata, tstdata = split_by_month(mydata, rating_only, keepratio, i_drop_ratio)
# trndata, vlddata, tstdata = split_by_month(mydata, isdense, isoverlap, isnew_amazon, i_drop_ratio)

try:
    dname = fn.split('_')[1].lower()
    dname = dname.split('.')[0]
except:
    dname = fn

if isdense == True:
    dname = dname + 'dense'
    
if i_drop_ratio > 0:
    dname = dname + '_dr' + str(i_drop_ratio)
    
basename = dname
if not os.path.exists(basename): os.makedirs(basename)
    
if keepratio != 1.0:
    dname = dname + str(keepratio)
    
dirname = dname+'/split/'
if not os.path.exists(dirname): os.makedirs(dirname)
    

    
# Save the dataset in csv format
writer = csv.writer(open(dirname+'trn.csv', 'w'))
writer.writerows(trndata)

writer = csv.writer(open(dirname+'vld.csv', 'w'))
writer.writerows(vlddata)

writer = csv.writer(open(dirname+'tst.csv', 'w'))
writer.writerows(tstdata)

print('\nDone\n')

