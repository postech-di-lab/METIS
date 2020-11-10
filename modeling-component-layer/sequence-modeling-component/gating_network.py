import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HGN(nn.Module):
    def __init__(self, num_users, num_items, args):
        
        super(HGN, self).__init__()
        self.args = args
        # init args
        L = args.sequence_length
        dims = args.dims

        self.U = nn.Embedding(num_users, dims)
        self.E = nn.Embedding(num_items, dims, padding_idx=0)

        self.feature_gate_item = nn.Linear(dims, dims)
        self.feature_gate_user = nn.Linear(dims, dims)

        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).cuda()
        self.instance_gate_user = Variable(torch.zeros(dims, L).type(torch.FloatTensor), requires_grad=True).cuda()
        
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.Q = nn.Embedding(num_items, dims, padding_idx=0)
        self.Qb = nn.Embedding(num_items, 1, padding_idx=0)
        # weight initialization
        self.Q.weight.data.normal_(0, 1.0 / self.Q.embedding_dim)
        self.Qb.weight.data.zero_()

    def lookup_layer(self, item_seq_indices, user_indices, items_to_predict):
        
        return self.E(item_seq_indices), self.U(user_indices), self.Q(items_to_predict), self.Qb(items_to_predict)

    def feature_gating_layer(self, S_i, u_i):
        
        return S_i * torch.sigmoid(self.feature_gate_item( S_i ) + self.feature_gate_user(u_i).unsqueeze(1))

    def instance_gating_layer(self, featg_S_i, u_i):
        
        g_score = torch.sigmoid(torch.matmul(featg_S_i, self.instance_gate_item.unsqueeze(0)).squeeze()+ u_i.mm(self.instance_gate_user))
        instg_S_i = featg_S_i * g_score.unsqueeze(2)
        instg_S_i = torch.sum(instg_S_i, dim=1)
        instg_S_i = instg_S_i/ torch.sum(g_score, dim=1).unsqueeze(1)

        return instg_S_i

    def prediction_layer(self, S_i,u_i, instg_S_i, q_j,qb_j, for_pred=False):
        
        if for_pred:
            q_j = q_j.squeeze()
            qb_j = qb_j.squeeze()
            
            # long-term
            res = u_i.mm(q_j.t()) + qb_j
            
            # short-term
            res += instg_S_i.mm(q_j.t())
            
            # item-item
            rel_score = torch.matmul(S_i, q_j.t().unsqueeze(0))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
        else:
            # long-term
            res = torch.baddbmm(qb_j, q_j, u_i.unsqueeze(2)).squeeze()

            # short-term
            res += torch.bmm(instg_S_i.unsqueeze(1), q_j.permute(0, 2, 1)).squeeze()
            
            # item-item
            rel_score = S_i.bmm(q_j.permute(0, 2, 1))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
            
        return res
       
    def forward(self, item_seq_indices, user_indices, items_to_predict, for_pred=False):
       
        # embedding look-up 
        S_i,u_i,q_j,qb_j = self.lookup_layer(item_seq_indices, user_indices, items_to_predict)

        # feature gating
        featg_S_i = self.feature_gating_layer(S_i,u_i)

        # instance gating
        instg_S_i = self.instance_gating_layer(featg_S_i,u_i)
        
        #return preference scores
        return self.prediction_layer(S_i,u_i,instg_S_i,q_j,qb_j,for_pred)

