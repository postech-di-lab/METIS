import sys
import os
import keras
import random as rn
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from evaluate import evaluate
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)
K.tensorflow_backend.set_session(sess)
import pandas as pd
import math
from sklearn.utils import shuffle
import model as M
import time
from generateNegatives import getNegativeSamples
from TimePreprocessor import timestamp_processor

embedding_size = 32
batch_size = 256
learning_rate = 0.001
patience = 10

sequence_length = 5
width = 128
depth = 4
dropout_rate = 0.1

model_name = "saved_model.h5"

tr_dataset = pd.read_csv("movielens/train.txt",sep=',',names="user_id,item_id,rating,timestamp".split(",")) 
va_dataset = pd.read_csv("movielens/validation.txt",sep=',',names="user_id,item_id,rating,timestamp".split(","))
te_dataset = pd.read_csv("movielens/test.txt",sep=',',names="user_id,item_id,rating,timestamp".split(","))

userSortedTimestamp = {}
for uid in tr_dataset.user_id.unique().tolist():
    trPosInstance = tr_dataset.loc[tr_dataset['user_id'] == uid]
    temp = va_dataset.loc[va_dataset['user_id'] == uid]
    vaPosInstance = temp.loc[temp['rating'] == 1]

    temp = te_dataset.loc[te_dataset['user_id'] == uid]
    tePosInstance = temp.loc[temp['rating'] == 1]

    posInstance = pd.concat([trPosInstance, vaPosInstance, tePosInstance], ignore_index=True)
    userSortedTimestamp[uid] = posInstance.sort_values(by=['timestamp'])

tr_dataset = timestamp_processor(tr_dataset, userSortedTimestamp, sequence_length)
va_dataset = timestamp_processor(va_dataset, userSortedTimestamp, sequence_length)
te_dataset = timestamp_processor(te_dataset, userSortedTimestamp, sequence_length)

num_users = max(tr_dataset['user_id'])
num_items = max(max(tr_dataset['item_id']), max(va_dataset['item_id']), max(te_dataset['item_id']))

model = M.TimelyRec([6], num_users, num_items, embedding_size, sequence_length, width, depth, dropout=dropout_rate)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate))

model.load_weights(model_name)

best_hr1 = 0
best_hr5 = 0
best_ndcg5 = 0
best_hr10 = 0
best_ndcg10 = 0
best_hr10_i = 0
best_hr10_i = 0 

# Evaluation
val_HR1, val_HR5, val_NDCG5, val_HR10, val_NDCG10 = evaluate(model, va_dataset, num_candidates=301, sequence_length=sequence_length)
test_HR1, test_HR5, test_NDCG5, test_HR10, test_NDCG10 = evaluate(model, te_dataset, num_candidates=301, sequence_length=sequence_length)

print ("Val")
print ("HR@1   : " + str(round(val_HR1, 4)))
print ("HR@5   : " + str(round(val_HR5, 4)))
print ("NDCG@5 : " + str(round(val_NDCG5, 4)))
print ("HR@10  : " + str(round(val_HR10, 4)))
print ("NDCG@10: " + str(round(val_NDCG10, 4)))
print ("")
    
print ("Test")
print ("HR@1   : " + str(round(test_HR1, 4)))
print ("HR@5   : " + str(round(test_HR5, 4)))
print ("NDCG@5 : " + str(round(test_NDCG5, 4)))
print ("HR@10  : " + str(round(test_HR10, 4)))
print ("NDCG@10: " + str(round(test_NDCG10, 4)))
print ('')