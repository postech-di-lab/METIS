# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, config, graph):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = config['emb_dim']
        self.n_layers = config['layers']
        self.reg = config['reg']
        self.graph = graph
        
        # 초기 임베딩
        self.embedding_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.emb_dim)
        
        # 초기화 (Normal Initialization)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def forward(self):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        
        # GCN Propagation (No interaction, pure implementation)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
            
        # Multi-scale 합치기 (Mean)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def get_loss(self, users_emb, items_emb, users, pos_items, neg_items):
        # Positive / Negative Score
        pos_scores = torch.mul(users_emb[users], items_emb[pos_items]).sum(dim=1)
        neg_scores = torch.mul(users_emb[users], items_emb[neg_items]).sum(dim=1)
        
        # BPR Loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # L2 Regularization
        user_ego = self.embedding_user(users)
        pos_ego = self.embedding_item(pos_items)
        neg_ego = self.embedding_item(neg_items)
        
        reg_loss = (user_ego.norm(2).pow(2) + 
                    pos_ego.norm(2).pow(2) + 
                    neg_ego.norm(2).pow(2)) / 2
        
        return loss + self.reg * reg_loss / users.shape[0]
