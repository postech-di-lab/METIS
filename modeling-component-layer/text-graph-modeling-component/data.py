import pandas as pd
import torch
from torch_geometric.data import Data


def AMT(device):
    df = pd.read_json('data/amt_preprocessed.json')
    num_users = df['user'].max().item() + 1
    num_items = df['item'].max().item() + 1
    df = pd.read_json('data/amt_train.json')
    num_ratings = len(df)

    x_s = torch.arange(num_users, device=device).reshape(-1, 1)
    x_t = torch.arange(num_items, device=device).reshape(-1, 1)
    edge_index = torch.tensor([df['user'].tolist(), df['item'].tolist()], device=device)
    edge_type = torch.tensor(df['rating'].tolist(), device=device)
    edge_attr = torch.load('data/review_preprocessed.pt', map_location=device)

    data_train = Data(x_s=x_s, x_t=x_t, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,
                      size=(num_users, num_items), train=True)

    df = pd.read_json('data/amt_test.json')
    edge_index = torch.tensor([df['user'].tolist(), df['item'].tolist()], device=device)
    edge_type = torch.tensor(df['rating'].tolist(), device=device)

    data_test = Data(x_s=x_s, x_t=x_t, edge_index=edge_index, edge_type=edge_type,
                     size=(num_users, num_items), train=False)

    return data_train, data_test
