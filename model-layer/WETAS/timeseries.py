import os
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# This class provides time-series with weak anomaly labels and dense anomaly labels
 
class TimeSeriesWithAnomalies(Dataset):
    def __init__(self, data_dir, split_size, split, **kwargs): 
        super().__init__()
        self.data_dir = data_dir
        self.split_size = split_size
        self._load_data(data_dir, split)

    def _load_data(self, data_dir, split):
        
        data, dlabel, wlabel = [], [], []
        filenames = os.listdir(os.path.join(data_dir, split))
        for filename in filenames:
            filepath = os.path.join(data_dir, split, filename)
            data_, label_, (length, input_size) = np.load(filepath, allow_pickle=True)
            data_, dlabel_, wlabel_ = self._preprocess(data_, label_, self.split_size)
            
            data.append(data_)
            dlabel.append(dlabel_)
            wlabel.append(wlabel_)

        self.input_size = input_size
        self.data = torch.cat(data, dim=0)
        self.dlabel = torch.cat(dlabel, dim=0)
        self.wlabel = torch.cat(wlabel, dim=0)

    def _preprocess(self, data, label, split_size):

        # normalize
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # split
        data = torch.Tensor(data)
        data = f.pad(data, (0, 0, split_size - data.shape[0] % split_size, 0), 'constant', 0)
        data = torch.unsqueeze(data, dim=0)
        data = torch.cat(torch.split(data, split_size, dim=1), dim=0)

        label = torch.Tensor(label)
        label = f.pad(label, (split_size - label.shape[0] % split_size, 0), 'constant', 0)
        label = torch.unsqueeze(label, dim=0)
        label = torch.cat(torch.split(label, split_size, dim=1), dim=0)
        
        dlabel = label
        wlabel = torch.max(label, dim=1)[0]
        
        return data, dlabel, wlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'dlabel': self.dlabel[idx],
            'wlabel': self.wlabel[idx]
        } 

