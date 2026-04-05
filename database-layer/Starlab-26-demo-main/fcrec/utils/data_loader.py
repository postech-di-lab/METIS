import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import numpy as np


class Pos_Neg_Sampler(Dataset): 
    '''
        for 1 user
    '''
    def __init__(self, args, rating_mat):
        self.args = args
        self.num_item = args.num_item
        self.num_ns = args.num_ns
        self.pos_items = list(rating_mat.keys())
        self.neg_items = []
        self.data = []
        
        for i in self.pos_items:
            self.data.append([i, 1.0])
            
        self.negative_sampling()
        
    def negative_sampling(self):
        sample_list = np.random.choice(list(range(self.num_item)), size=10 * self.num_ns * len(self.pos_items))
        sample_idx = 0
        
        for item in self.pos_items:
            ns_count = 0
            
            while True:
                neg_item = sample_list[sample_idx]
                sample_idx += 1
                
                if not neg_item in self.pos_items:
                    self.data.append([neg_item, 0.0])
                    self.neg_items.append(neg_item)
                    ns_count += 1
                if ns_count == self.num_ns:
                    break
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
