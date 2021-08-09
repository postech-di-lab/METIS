import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def EuclideanDistance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))


def squaredEuclideanDistance(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def summarize_vectors(vectors, degrees):
    normalized_mean_vector = F.normalize(torch.matmul(degrees, vectors), p=2, dim=-1)
    mean_norm = torch.matmul(degrees, torch.norm(vectors, dim=-1).unsqueeze(-1))
    summary_vector = normalized_mean_vector * mean_norm

    return summary_vector


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn[:, :, 0] = -1e+38
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        nn.init.normal_(self.w_q.weight, 0, 0.01)
        nn.init.normal_(self.w_k.weight, 0, 0.01)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), dropout=dropout)

    def forward(self, q, k, v, mask=None):
        q = torch.relu(self.w_q(q))
        k = torch.relu(self.w_k(k))
        output = self.attention(q, k, v, mask=mask)  
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()
        self.WS1 = nn.Linear(d_in, d_hid)
        self.WS2 = nn.Linear(d_hid, d_in)
        nn.init.normal_(self.WS1.weight.data, 0.0, 0.01)
        nn.init.normal_(self.WS2.weight.data, 0.0, 0.01)

    def forward(self, x):
        output = self.WS2(torch.relu(self.WS1(x)))
        return output


class SelfAttentionNetwork(nn.Module):
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.1):
        super(SelfAttentionNetwork, self).__init__()
        self.slf_attn = SelfAttention(d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, session, non_pad_mask=None, slf_attn_mask=None):
        X = session
        AX = self.slf_attn(X, X, X, mask=slf_attn_mask)
        Z = X + self.dropout(AX)
        Z *= non_pad_mask

        fS = self.pos_ffn(Z)
        fS *= non_pad_mask

        return fS
