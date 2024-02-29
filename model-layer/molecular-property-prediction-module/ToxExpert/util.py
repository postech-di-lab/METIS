
import random
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans


class Scf_index:
    def __init__(self, dataset, args):
        self.device = args.device
        
        self.max_scf_idx = None
        self.scfIdx_to_label = None
        self.num_scf = None

        self.get_scf_idx(dataset)


    def get_scf_idx(self, dataset):
        ''''
        scf label: scf 에 속한 sample 수가 많은 순서부터 desending order 로 sorting 해서 label 매김
        self.num_train_scf = train set에 있는 scf 의 종류 
        self. 
        '''
        
        scf = defaultdict(int)
        max_scf_idx = 0 
        for data in dataset:
            idx = data.scf_idx.item()
            scf[idx] += 1
            if max_scf_idx < idx:
                max_scf_idx = idx
        self.max_scf_idx = max_scf_idx
        scf = sorted(scf.items(), key=lambda x: x[1], reverse=True)
        
        self.scfIdx_to_label = torch.ones(max_scf_idx+1).to(torch.long).to(torch.long) * -1
        self.scfIdx_to_label = self.scfIdx_to_label.to(self.device) 

        for i, k in enumerate(scf):
            self.scfIdx_to_label[k[0]] = i 

        self.num_scf = len(scf)


def load_models(args, model):
    
    if not args.ckpt_all == "":

        load = torch.load(args.ckpt_all)
        mis_keys, unexp_keys = model.load_state_dict(load, strict=False)
        print('missing_keys:', mis_keys)
        print('unexpected_keys:', unexp_keys)
    
    elif not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)



### utils for eval
def cal_roc(y_true, y_scores):
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
    
    return sum(roc_list) / (len(roc_list) + 1e-10) * 100



def init_centroid(model, z_s, num_experts):
    
    z_s_arr = z_s.detach().cpu().numpy()

    num_data = z_s_arr.shape[0]
    if  num_data> 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        z_s_arr = z_s_arr[mask_idx[:35000]]

    kmeans = KMeans(n_clusters=num_experts, random_state=0).fit(z_s_arr)
    centroids = kmeans.cluster_centers_

    model.cluster.data = torch.tensor(centroids).to(model.cluster.device)

def get_z(model, loader, device):
    model.train()
    
    z_s = []    
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            _, z, _ = model(batch)
        z_s.append(z)

    
    z_s = torch.cat(z_s, dim=0)
    return z_s


def set_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)