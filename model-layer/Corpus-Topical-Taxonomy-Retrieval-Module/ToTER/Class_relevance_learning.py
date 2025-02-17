## This is a simple implementation for class relevance learning.

import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm, trange
import numpy as np
import pickle

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim
        
def train_classifier():
    
    from Utils import train_dataset, Classifier
    
    PLM_corpus_embedding = np.load('resource/PLM_corpus_embedding.npy') # document embedding
    PLM_label_embedding = np.load('resource/PLM_label_embedding.npy')   # topic class embedding

    with open('resource/silver_labels', 'rb') as f:
        silver_labels = pickle.load(f)

    num_class = PLM_label_embedding.shape[0]
    mlb = MultiLabelBinarizer(classes=[i for i in range(num_class)], sparse_output=True)

    train_X, train_X2 = [], []
    train_Y = []

    for index in range(PLM_corpus_embedding.shape[0] // 2):
        if index not in silver_labels: continue
        train_X.append(PLM_corpus_embedding[index])
        train_X2.append(PLM_corpus_embedding[(PLM_corpus_embedding.shape[0] // 2) + index])
        train_Y.append(silver_labels[index])

    train_X, train_X2 = np.asarray(train_X), np.asarray(train_X2)
    train_X = np.concatenate([train_X, train_X2, PLM_label_embedding], 0)

    raw_train_Y = train_Y + train_Y + [[i] for i in range(PLM_label_embedding.shape[0])]
    train_Y = mlb.fit_transform(raw_train_Y)

    train_X = torch.FloatTensor(train_X)
    label_embedding = torch.FloatTensor(PLM_label_embedding).to('cuda')

    train_dataset = train_dataset(train_X, train_Y)
    train_loader = data.DataLoader(train_dataset, batch_size=10240, shuffle=True)

    model = Classifier(num_class).to('cuda')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(1000):

        epoch_loss = 0
        for _, mini_batch in enumerate(tqdm(train_loader)):

            batch_indices, batch_X = mini_batch
            batch_X = batch_X.to('cuda')
            batch_Y = train_loader.dataset.get_labels(batch_indices)
            batch_Y = torch.FloatTensor(batch_Y).to('cuda')

            batch_output = model(batch_X)
            batch_loss = criterion(batch_output, batch_Y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
        print('Epoch: {}, Loss: {:.3f}'.format(epoch, epoch_loss))

    print('Train done!')
    torch.save(model.state_dict(), 'resource/Classifier_model')