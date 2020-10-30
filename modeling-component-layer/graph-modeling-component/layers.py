import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_feature, num_class):
        super(FCN, self).__init__()

        self.feature_extractor = nn.Sequential(
                                    nn.Linear(num_feature, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, num_class)
                                )

    def forward(self, x):
        return self.feature_extractor(x)
    
    
    
class GCN_layer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, X):
        return self.fc(torch.spmm(self.A, X)) #이웃 정보 종합

class GCN(nn.Module):
    def __init__(self, num_feature, num_class, A):
        super(GCN, self).__init__()

        self.feature_extractor = nn.Sequential(
                                    GCN_layer(num_feature, 10, A),
                                    nn.ReLU(),
                                    GCN_layer(10, num_class, A)
                                )
        
    def forward(self, X):
        return self.feature_extractor(X)
    
    

class GAT_layer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, A, dropout, is_final=False):
        super(GAT_layer, self).__init__()
        
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.is_final = is_final
        
        # for higher level feature
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        
    def forward(self, X):
        
        H = torch.mm(X, self.W)
        N = H.size(0)
        
        # calculate attention scores
        a_input = torch.cat([H.repeat(1, N).view(N * N, -1), H.repeat(N, 1)], dim=-1).view(N, N, self.out_features * 2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # masked attention
        zero_vec = -9e15*torch.ones_like(e)
        masked_e = torch.where(self.A > 0, e, zero_vec)
        alpha = F.softmax(masked_e, dim=-1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        
        # apply attention
        H_new = torch.mm(alpha, H)
        
        if self.is_final:
            return H_new
        else:
            return F.elu(H_new)
        
class GAT(nn.Module):
    def __init__(self, num_feature, num_class, A, dropout, num_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.GAT_layer = [GAT_layer(num_feature, 8, A, dropout, False) for _ in range(num_heads)]
        for idx, GAT_head in enumerate(self.GAT_layer):
            self.add_module('att_' + str(idx), GAT_head)
        
        self.out_layer = GAT_layer(8 * num_heads, num_class, A, dropout, True)
        
        
    def forward(self, X):
        H = F.dropout(X, self.dropout, training=self.training)
        # GAT layer 1
        H = torch.cat([GAT_head(X) for GAT_head in self.GAT_layer], dim=-1)
        H = F.dropout(H, self.dropout, training=self.training)
        
        # GAT layer 2
        H = self.out_layer(H)
        H = F.elu(H)
        return H
