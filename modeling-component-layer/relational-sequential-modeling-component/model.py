import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import random
import numpy as np

class SASREC_ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(SASREC_ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class SASREC_MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(SASREC_MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = SASREC_ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        # output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)

        return output, attn


class SASREC_PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(SASREC_PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        # output = self.dropout(output)
        # output = self.layer_norm(output + residual)
        return output

class SASREC_SelfAttentionBlock(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SASREC_SelfAttentionBlock, self).__init__()
        self.slf_attn = SASREC_MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = SASREC_PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        residual = enc_input
        enc_input = self.layer_norm(enc_input)

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = residual + self.dropout(enc_output)
        # enc_output = self.dropout(enc_output)
        enc_output *= non_pad_mask

        # residual = enc_output
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        enc_output = residual + self.dropout(enc_output)

        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, num_position, num_head, num_layers, dropout):
        super(SASRec, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_position = num_position
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx = 0)
        self.positional_embedding = nn.Embedding(self.num_position, self.embedding_dim)
        
        self.layer_stack = nn.ModuleList([
            SASREC_SelfAttentionBlock(self.embedding_dim, self.embedding_dim, self.num_head, self.embedding_dim, self.embedding_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, session):
        # session: batch_size * max_length
        
        batch_size = session.size()[0]
        max_length = session.size()[1]

        isValid = (session != 0.0).to(torch.float) # batch_size * max_length

        # Build embedding
        itemEmbeddings = self.item_embedding((torch.arange(self.num_items+1)).to(self.device)) # num_items * dim
        sessionEmbeddings = self.item_embedding(session) # batch_size * max_length * dim

        ##### SASRec
        # print torch.arange(session.shape[1])
        # print torch.arange(session.shape[1]).to(self.device)
        # print self.num_position - session.shape[1]
        # print torch.arange(session.shape[1]).to(self.device) + (self.num_position - session.shape[1])
        positionalEmbeddings = self.positional_embedding(torch.arange(session.shape[1]).to(self.device) + (self.num_position - session.shape[1]))

        # session representation
        sessionRepresentations = sessionEmbeddings + positionalEmbeddings # batch_size * max_length * dim
        sessionRepresentations *= isValid.unsqueeze(2)

        # mask
        subsequent_mask = torch.triu(torch.ones((max_length, max_length), device=self.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1) # batch_size * max_length * max_length
        padding_mask = session.eq(0) # batch_size * max_length
        padding_mask = padding_mask.unsqueeze(1).expand(-1, max_length, -1) # batch_size * max_length * max_length
        mask = (subsequent_mask + padding_mask).gt(0)
        mask[:, :, 0] = 0
        non_pad_mask = session.ne(0).type(torch.float).unsqueeze(-1)
        # mask = torch.zeros_like(mask).to(self.device)
        # non_pad_mask = torch.zeros_like(non_pad_mask).to(self.device)
        # enc_output = self.emb_dropout(sessionRepresentations)
        enc_output = sessionRepresentations # batch_size * max_length * dim
        item_emb = enc_output[:, -1, :] # batch_size * dim
        
        for enc_layer in self.layer_stack:
            enc_output, enc_self_attention = enc_layer(
                enc_output, non_pad_mask = non_pad_mask, slf_attn_mask=mask)

        sessionRepresentation = enc_output[:, -1, :] # batch_size * dim

        output = torch.matmul(sessionRepresentation, itemEmbeddings.t()) # batch_size * num_items
        
        return output