import math
import argparse


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from splitters import data_split
from loader import MoleculeDataset
from model import GNN_topexpert
from util import *

def load_args():
    parser = argparse.ArgumentParser()

# seed & device
    parser.add_argument('--device_no', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
   
#dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")

#model
    parser.add_argument('-i', '--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('-c', '--ckpt_all', type=str, default='',
                        help='filename to read the model ')
    

    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
    parser.add_argument('--gnn_type', type=str, default="gin")


# train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')


#optimizer
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

## loss balance
    parser.add_argument('--alpha', type=float, default=0.1, help="balance parameter for clustering")
    parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")

## clustering
    parser.add_argument('--min_temp', type=float, default=1, help=" temperature for gumble softmax, annealing")
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--gate_dim', type=int, default=50, help="gate embedding space dimension, 50 or 300")


    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")

    # Bunch of classification tasks
    if args.dataset == "tox21":
        args.num_tasks = 12
        args.num_classes = 2
    elif args.dataset == "hiv":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "pcba":
        args.num_tasks = 128
        args.num_classes = 2
    elif args.dataset == "muv":
        args.num_tasks = 17
        args.num_classes = 2
    elif args.dataset == "bace":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "bbbp":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "toxcast":
        args.num_tasks = 617
        args.num_classes = 2
    elif args.dataset == "sider":
        args.num_tasks = 27
        args.num_classes = 2
    elif args.dataset == "clintox":
        args.num_tasks = 2
        args.num_classes = 2
    else:
        raise ValueError("Invalid dataset name.")

    return args


def train(args, model, loader, optimizer, scf_class):
    model.train()
    
    for batch in loader:

        model.T = max(torch.tensor(args.min_temp), model.T * args.temp_alpha)
        
        batch = batch.to(args.device)
        num_graph = batch.id.shape[0]  
        labels = batch.y.view(num_graph, -1).to(torch.float64)
        
        _, _,  temp_q = model(batch)
        temp_q = temp_q.data

        p = model.target_distribution(temp_q) 
        
        clf_logit, z, q = model(batch)
        g, q_idx = model.assign_head(q) # g--> N x tasks x head
        
        scf_idx = scf_class.scfIdx_to_label[batch.scf_idx] 

        clf_loss_mat, num_valid_mat = model.clf_loss(clf_logit, labels, g)
        classification_loss = torch.sum(clf_loss_mat/num_valid_mat)/args.num_tasks
 
        cluster_loss =  F.kl_div(q.log(), p, reduction='sum')
        align_loss = model.alignment_loss(scf_idx, q)
        
        loss_total = classification_loss  + args.alpha * (cluster_loss / num_graph) + args.beta * align_loss
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()


def eval(args, model, loader):
    model.eval()
    
    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)
        with torch.no_grad():
            clf_logit, z, q_origin = model(batch)
            q, q_idx = model.assign_head(q_origin) # N x tasks x head
            scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)             
        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores)    

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    avg_roc = cal_roc(y_true, y_scores)

    return  avg_roc


def main(args):
    set_seed(args.seed)    

    # dataset split & data loader
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    ## criterion
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    ## set additional args
    scf_tr = Scf_index(train_dataset, args)
    args.num_tr_scf = scf_tr.num_scf
    
    num_iter = math.ceil(len(train_dataset) / args.batch_size) 
    args.temp_alpha = np.exp(np.log(args.min_temp / 10 + 1e-10) / (args.epochs * num_iter))


    ## define a model and load chek points
    model = GNN_topexpert(args, criterion)
    load_models(args, model)
    model = model.to(args.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)


    ###### init centroid using randomly initialized gnn
    zs_init = get_z(model, train_loader, args.device)
    init_centroid(model, zs_init, args.num_experts)
 
    for epoch in range(1, args.epochs + 1):

        train(args, model, train_loader, optimizer, scf_tr)

        val_acc = eval(args, model, val_loader)
        te_acc = eval(args, model, test_loader)

        print(f'{epoch}epoch, val acc:{val_acc:.1f}, test acc:{te_acc:.1f} ')

if __name__ == "__main__":
    args = load_args()
    
    
    main(args)


   
   

