import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class GNN(nn.Module):
    def __init__(self, num_users, num_items, embed_size, dropout):
        super(GNN, self).__init__()
        self.embed_user = nn.Embedding(num_users, embed_size)
        self.embed_item = nn.Embedding(num_items, embed_size)
        self.conv_user = Conv(True, embed_size, dropout)
        self.conv_item = Conv(False, embed_size, dropout)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_size, 1), nn.Sigmoid(),
        )

        self.x_s_conved = None
        self.x_t_conved = None

    def forward(self, data):
        if data.train:
            x_s = self.embed_user(data.x_s.reshape(-1))
            x_t = self.embed_item(data.x_t.reshape(-1))

            self.x_s_conved = F.relu(
                self.conv_user(x_s, x_t, data.edge_index, data.edge_type, data.edge_attr, data.size))
            self.x_t_conved = F.relu(
                self.conv_item(x_s, x_t, data.edge_index, data.edge_type, data.edge_attr, data.size))

        x_i = self.x_s_conved.index_select(0, data.edge_index[0])
        x_j = self.x_t_conved.index_select(0, data.edge_index[1])
        x = torch.cat((x_i, x_j), 1)
        return self.regressor(x) * 4 + 1


class Conv(MessagePassing):
    def __init__(self, reverse, embed_size, dropout):
        super(Conv, self).__init__(aggr='mean')
        self.reverse = reverse
        self.fc_i = nn.Linear(embed_size, embed_size)
        self.fc_j = nn.Linear(embed_size, embed_size * 5)
        self.fc_e = nn.Sequential(
            nn.Linear(768, embed_size * 5), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_size * 5, embed_size), nn.ReLU()
        )

    def forward(self, x_s, x_t, edge_index, edge_type, edge_attr, size):
        if self.reverse:
            edge_index = edge_index.flip(0)
            size = size[::-1]
            x = (self.fc_j(x_t), x_s)
        else:
            x = (self.fc_j(x_s), x_t)

        return self.propagate(edge_index=edge_index, size=size, x=x,
                              edge_type=edge_type, edge_attr=edge_attr)

    def message(self, x_j, edge_type, edge_attr):
        message_node = x_j.reshape(len(x_j), 5, -1)[torch.arange(len(x_j)), edge_type - 1]
        message_edge = self.fc_e(edge_attr)
        return message_node + message_edge

    def update(self, inputs, x):
        return self.fc_i(x[1]) + inputs
