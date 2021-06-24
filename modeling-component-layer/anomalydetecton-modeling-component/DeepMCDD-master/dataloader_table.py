import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_table_data(batch_size, data_dir, dataset, oodclass_idx, fold_idx, **kwargs):

    data_path = os.path.join(data_dir, dataset + '_preproc.npy')
    features, labels, num_classes = np.load(data_path, allow_pickle=True)
    n_data = len(labels)

    id_classes = [c for c in range(num_classes) if c != oodclass_idx]
    id_indices = {c: [i for i in range(len(labels)) if labels[i] == c] for c in id_classes}

    np.random.seed(0)
    for c in id_classes:
        np.random.shuffle(id_indices[c])
    test_id_indices = {c: id_indices[c][int(0.2*fold_idx*len(id_indices[c])):int(0.2*(fold_idx+1)*len(id_indices[c]))] for c in id_classes}

    id_indices = np.concatenate(list(id_indices.values()))
    test_id_indices = np.concatenate(list(test_id_indices.values()))
    train_id_indices = np.array([i for i in id_indices if i not in test_id_indices])
    ood_indices = [i for i in range(len(labels)) if labels[i] == oodclass_idx]

    for i in range(len(labels)):
        if labels[i] == oodclass_idx:
            labels[i] = -1
        elif labels[i] > oodclass_idx:
            labels[i] -= 1

    train_dataset = TensorDataset(torch.Tensor(features[train_id_indices]), 
                                        torch.LongTensor(labels[train_id_indices]))
    test_id_dataset = TensorDataset(torch.Tensor(features[test_id_indices]), 
                                        torch.LongTensor(labels[test_id_indices]))
    test_ood_dataset = TensorDataset(torch.Tensor(features[ood_indices]), 
                                        torch.LongTensor(labels[ood_indices]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_id_loader = DataLoader(dataset=test_id_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_ood_loader = DataLoader(dataset=test_ood_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_id_loader, test_ood_loader

def get_total_table_data(batch_size, data_dir, dataset, fold_idx, **kwargs):

    data_path = os.path.join(data_dir, dataset + '_preproc.npy')
    features, labels, num_classes = np.load(data_path, allow_pickle=True)
    n_data = len(labels)
    id_indices = [i for i in range(len(labels))] 

    np.random.seed(0)
    np.random.shuffle(id_indices)
    test_id_indices = id_indices[int(0.2*fold_idx*len(id_indices)):int(0.2*(fold_idx+1)*len(id_indices))]
    train_id_indices = np.array([i for i in id_indices if i not in test_id_indices])

    test_dataset = TensorDataset(torch.Tensor(features[test_id_indices]), 
                                        torch.LongTensor(labels[test_id_indices]))
    train_dataset = TensorDataset(torch.Tensor(features[train_id_indices]), 
                                        torch.LongTensor(labels[train_id_indices]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loader

