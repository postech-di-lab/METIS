import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TripletDataset(Dataset):
    """Dataset for BPR model.

    BPR model require triplet (u, i, j) which represent user,
    positive item and negative item, respectively.

    Args:
        S_train (list of list): training pair 
        item_size (int): total number of item.
    """
    def __init__(self, S_train, item_size):
        self.S_train = S_train
        self.item_size = item_size

    def __getitem__(self, idx):
        """Create triplet using sampling."""
        u = np.random.choice(range(len(S_train)), size=1)[0]
        i = np.random.choice(S_train[u], size=1)[0]
        j = np.random.choice(range(item_size), size=1)[0]
        while j in S_train[u]:
            j = np.random.choice(range(item_size), size=1)[0]
        return u, i, j

    def __len__(self):
        return len(self.S_train)


class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization

    https://arxiv.org/pdf/1205.2618

    Args:
        user_size (int): total number of user (indexed by id)
        item_size (int): total number of item (indexed by id)
        dim (int): dimension for matrix factorization
        reg_user (float): regularization parameter for user
        reg_pos_item (float): regularization parameter for positive item
        reg_neg_item (float): regularization parameter for negative item
        reg_bias (float): regularization parameter for bias
    """
    def __init__(self, user_size, item_size, dim,
                reg_user, reg_pos_item, reg_neg_item, reg_bias):
        super().__init__()
        self.W = nn.Parameter(torch.rand(user_size, dim))
        self.H = nn.Parameter(torch.rand(item_size, dim))
        self.B = nn.Parameter(torch.rand(item_size))
        self.reg_user = reg_user
        self.reg_pos_item = reg_pos_item
        self.reg_neg_item = reg_neg_item
        self.reg_bias = reg_bias

    def forward(self, u, i, j):
        """Compute the BPR-OPT to optimize.
        
        Args:
            u (tensor.long[N]): user index
            i (tensor.long[N]): positive item index
            j (tensor.long[N]): negative item index
        Returns:
            BPR-OPT (tensor.long[1])
        """
        x_ui = torch.mul(self.W[u,:], self.H[i,:]).sum(dim=1)
        x_uj = torch.mul(self.W[u,:], self.H[j,:]).sum(dim=1)
        x_uij = x_ui - x_uj + self.B[i] - self.B[j]
        log_prob = torch.log(torch.sigmoid(x_uij)).mean()
        log_prob -= self.reg_user * torch.norm(self.W[u,:], p=2, dim=1).mean()
        log_prob -= self.reg_pos_item * torch.norm(self.H[i,:], p=2, dim=1).mean()
        log_prob -= self.reg_neg_item * torch.norm(self.H[j,:], p=2, dim=1).mean()
        log_prob -= self.reg_bias * self.B[i].mean()
        log_prob -= self.reg_bias * self.B[j].mean()
        return -log_prob

    def predict(self, u, i):
        """Predict the preference using user and item
        
        Args:
            u (tensor.long[N]): user index
            i (tensor.long[N]): positive item index
        Returns:
            preference value (tensor.float[N])
        """
        return torch.mul(self.W[u,:], self.H[i,:]).sum(dim=1)

