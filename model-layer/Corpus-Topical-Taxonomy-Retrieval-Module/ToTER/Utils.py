import numpy as np
import pickle

from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data

class train_dataset(data.Dataset):
    def __init__(self, X, Y):
        super(train_dataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx]

    def get_labels(self, batch_indices):
        return self.Y[batch_indices].todense()
    
class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.text_f = nn.Sequential(nn.Linear(768, (768+num_class)//2), nn.ReLU(), \
                                    nn.Linear((768+num_class)//2, num_class))

    def forward(self, batch_X):
        return self.text_f(batch_X)

def print_metric(metrics):
    print('NDCG@10:', metrics[0]['NDCG@10'], ', NDCG@100:', metrics[0]['NDCG@100'])
    print('Recall@100:', metrics[2]['Recall@100'], ', Recall@500:', metrics[2]['Recall@500'], \
          ', Recall@1000:', metrics[2]['Recall@1000'], ', Recall@2500:', metrics[2]['Recall@2500'])

def to_np(x):
    return x.data.cpu().numpy()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(x):
    x /= np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    return x

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def score_mat_2_rank_mat(score_mat):
    rank_tmp = np.argsort(-score_mat, axis=-1)
    rank_mat = np.zeros_like(rank_tmp)
    for i in range(rank_mat.shape[0]):
        row = rank_tmp[i]
        rank_mat[i][row] = torch.LongTensor(np.arange(len(row)))
    return rank_mat

def eval_full_score_mat(score_mat, q_id_list, c_id_list):
    results_dict = {}
    for row in tqdm(range(score_mat.shape[0])):
        q_id = q_id_list[row]
        if q_id not in results_dict: results_dict[q_id] = {}
        for col in range(score_mat.shape[1]):
            c_id = c_id_list[col]
            results_dict[q_id][c_id] = float(score_mat[row][col])
    return results_dict

def return_topK_result(result_dict, topk=1000):
    new_results = {}
    for qid in result_dict:
        new_results[qid] = {}
        for pid, score in sorted(result_dict[qid].items(), key=lambda item: item[1], reverse=True)[:topk]:
            new_results[qid][pid] = score
    return new_results

def convert_to_rank_score(org_results, raw=0.05):
    length = len(org_results[list(org_results.keys())[0]])
    rank_importance = np.asarray([(1 / rank) ** (raw) for rank in range(1, length + 1)])
    new_top_result = {}
    
    for qid in tqdm(org_results):
        new_top_result[qid] = {}
        tmp = np.asarray(list(org_results[qid].items()))
        top_items, scores = tmp[:,0], tmp[:,1].astype(float)
        rank_tmp = np.argsort(-scores)
        
        rank_score = np.zeros_like(rank_tmp).astype(float)
        rank_score[rank_tmp] = rank_importance[:rank_tmp.shape[0]]
        
        for idx in range(top_items.shape[0]):
            new_top_result[qid][top_items[idx]] = rank_score[idx]
            
    return new_top_result

    
def filtering(score_mat, quantile=0.6, level_list=None, ignore_level_list=[]):
    score_mat_copy = score_mat.copy()
    
    with open('resource/level2id_dict', 'rb') as f:
        level2id_dict = pickle.load(f) 
    
    if level_list == None:
        level_list = level2id_dict.keys()
    
    for level in tqdm(level_list):
        if level in ignore_level_list:
            quantile = 1.0
        level_node_list = np.asarray(list(level2id_dict[level]))
        th_mat = np.quantile(score_mat[:,level_node_list], quantile, axis=1, keepdims=True)
        
        tmp = np.zeros_like(score_mat_copy)
        tmp[:, level_node_list] = 1
        condition = (tmp * True) * (score_mat_copy <= th_mat)

        score_mat_copy[condition == True] = 0.
    return score_mat_copy    
    
def compute_topical_relatedness(filtered_X, filtered_Y):

    rank_X = score_mat_2_rank_mat(filtered_X)
    rank_Y = score_mat_2_rank_mat(filtered_Y)

    input_X = (1 / (rank_X + 1)) ** 0.05
    input_Y = (1 / (rank_Y + 1)) ** 0.05

    score_mat = np.matmul(input_X, input_Y.T)
    return score_mat

def z_normalize(results):
    new_results = {}
    mean = np.asarray([[results[qid][cid] for cid in results[qid]] for qid in results]).mean()
    std = np.asarray([[results[qid][cid] for cid in results[qid]] for qid in results]).std()    
    
    for qid in results:
        new_results[qid] = {}
        for cid in results[qid]:
            new_results[qid][cid] = (results[qid][cid] - mean) / std
    return new_results