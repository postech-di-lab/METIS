import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

import os, time
import argparse
import copy

from model import DilatedCNN, ResidualBlock
from timeseries import TimeSeriesWithAnomalies
from softdtw_cuda import SoftDTW
from utils import * 

def evaluation(args, model, data_loader, bce, epoch, train_loader=None):
    model.eval()
    total, total_loss, total_bce_loss, total_dtw_loss = 0, 0, 0, 0
    wscores, wlabels = [], []
    dscores, dlabels = [], []
    wresult, dresult = np.zeros(3), np.zeros(3)

    for itr, batch in enumerate(data_loader):
        data = batch['data'].cuda()
        wlabel = batch['wlabel'].cuda()
        dlabel = batch['dlabel'].cuda()
        
        with torch.no_grad():
            out = model.get_scores(data)
            bce_loss = bce(out['wscore'], wlabel)
            dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)
            loss = bce_loss + dtw_loss

            total += data.size(0)
            total_bce_loss += bce_loss.item() * data.size(0)
            total_dtw_loss += dtw_loss.item() * data.size(0)
            total_loss += loss.item() * data.size(0)

            # weak prediction
            wresult += compute_wacc(out['wpred'], wlabel)
            wscores.append(out['wscore'])
            wlabels.append(wlabel)

            # dense prediction
            dpred = model.get_dpred(out['output'], out['wpred'])
            dresult += compute_dacc(dpred, dlabel)
            dscores.append(out['dscore'])
            dlabels.append(dlabel)

    if train_loader is not None:
        for itr, batch in enumerate(train_loader):
            data = batch['data'].cuda()
            wlabel = batch['wlabel'].cuda()
            
            with torch.no_grad():
                out = model.get_scores(data)
                wscores.append(out['wscore'])
                wlabels.append(wlabel)

    ret = {}
    ret['loss'] = total_loss / total
    ret['bce_loss'] = total_bce_loss / total
    ret['dtw_loss'] = total_dtw_loss / total
    
    ret['wprecision'], ret['wrecall'], ret['wf1'], ret['wIoU'] = compute_precision_recall(wresult)
    ret['dprecision'], ret['drecall'], ret['df1'], ret['dIoU'] = compute_precision_recall(dresult)

    wscores, wlabels = torch.cat(wscores, dim=0), torch.cat(wlabels, dim=0)
    dscores, dlabels = torch.cat(dscores, dim=0), torch.cat(dlabels, dim=0)

    # Weak Result
    ret['wauc'] = compute_auc(wscores, wlabels)
    ret['wauprc'] = compute_auprc(wscores, wlabels)
    ret['wbestf1'], ret['global_threshold'] = compute_bestf1(wscores, wlabels, return_threshold=True)
    wbestpred = (wscores >= ret['global_threshold']).type(torch.cuda.FloatTensor)
    wbestresult = compute_dacc(wbestpred, wlabels)
    ret['wbprecision'], ret['wbrecall'], ret['wbf1'], ret['wbIoU'] = compute_precision_recall(wbestresult)

    # Dense Result
    ret['dauc'] = compute_auc(dscores, dlabels)
    ret['dauprc'] = compute_auprc(dscores, dlabels)
    ret['dbestf1'], ret['local_threshold'] = compute_bestf1(dscores, dlabels, return_threshold=True)
    dbestpred = (dscores >= ret['local_threshold']).type(torch.cuda.FloatTensor)
    dbestresult = compute_dacc(dbestpred, dlabels)
    ret['dbprecision'], ret['dbrecall'], ret['dbf1'], ret['dbIoU'] = compute_precision_recall(wbestresult)
    return ret


def train_bceloss(args, train_dataset, valid_dataset, test_dataset):

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtw = SoftDTW(use_cuda=True, gamma=args.gamma, normalize=False)

    model = DilatedCNN(input_size=train_dataset.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                kernel_size=args.kernel_size,
                n_layers=args.n_layers,
                pooling_type=args.pooling_type,
                local_threshold=args.local_threshold,
                granularity=args.granularity,
                beta=args.beta,
                split_size=args.split_size,
                dtw=dtw)
    model.cuda()

    bce = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start Training
    best_valid_f1, best_valid_loss = -np.inf, np.inf
    best_state, no_imprv_cnt = None, 0
    for epoch in range(args.n_epochs):
        model.train()
        total_step = len(train_loader)
        total, total_loss, total_bce_loss, total_dtw_loss = 0, 0, 0, 0
        for itr, batch in enumerate(train_loader):
            data = batch['data'].cuda()
            wlabel = batch['wlabel'].cuda()

            out = model.get_scores(data)
            bce_loss = bce(out['wscore'], wlabel)
            dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)
            loss = bce_loss + dtw_loss

            with torch.no_grad():
                total += data.size(0)
                total_bce_loss += bce_loss.item() * data.size(0)
                total_dtw_loss += dtw_loss.item() * data.size(0)
                total_loss += loss.item() * data.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = total_loss / total
        train_bce_loss = total_bce_loss / total
        train_dtw_loss = total_dtw_loss / total

        # Evaluate the model using the validation set
        valid_result = evaluation(args, model, valid_loader, bce, epoch, train_loader=train_loader) # to stably obtain the global_threshold
        
        print('Epoch [{}/{}], step [{}/{}], Train Loss : {:.6f} (BCE : {:.6f}, DTW : {:.6f}), Valid loss : {:.6f} (BCE : {:.6f}, DTW : {:.6f})'
                .format(epoch+1, args.n_epochs, itr+1, total_step, \
                train_loss, train_bce_loss, train_dtw_loss, \
                valid_result['loss'], valid_result['bce_loss'], valid_result['dtw_loss']))
        print('\tValid (WEAK) AUC : {:.6f}, AUPRC : {:.6f}, Best F1 : {:.6f},  Precision : {:.6f}, Recall : {:.6f}, threshold : {:.6f}'.format(
            valid_result['wauc'], valid_result['wauprc'], valid_result['wbestf1'], valid_result['wprecision'], valid_result['wrecall'], valid_result['global_threshold']))
       
        # Evaluate the model using the test set
        model.global_threshold = valid_result['global_threshold']
        test_result = evaluation(args, model, test_loader, bce, epoch)

        print('\tTest  (WEAK) AUC : {:.6f}, AUPRC : {:.6f}, Best F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}'.format(
            test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wprecision'], test_result['wrecall']))
        print('\tTest (DENSE) F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}, IoU : {:.6f}'.format(
            test_result['df1'], test_result['dprecision'], test_result['drecall'], test_result['dIoU']))

        # Check the condition for the early-stopping
        if args.stopping == 'f1':
            if valid_result['wbestf1'] > best_valid_f1:
                best_valid_f1 = valid_result['wbestf1']
                best_state = copy.deepcopy(model.state_dict())
                no_imprv_cnt = 0
            elif no_imprv_cnt >= args.patience:
                print('Early stop at {} epochs with valid f1 {:.6f}'.format(epoch - args.patience, best_valid_f1))
                break
            else:
                no_imprv_cnt += 1
        
        elif args.stopping == 'loss':
            if valid_result['loss'] < best_valid_loss:
                best_valid_loss = valid_result['loss']
                best_state = copy.deepcopy(model.state_dict())
                no_imprv_cnt = 0
            elif no_imprv_cnt >= args.patience:
                print('Early stop at {} epochs with valid loss {:.6f}'.format(epoch - args.patience, best_valid_loss))
                break
            else:
                no_imprv_cnt += 1

        print('')

    model.load_state_dict(best_state)
    
    valid_result = evaluation(args, model, valid_loader, bce, epoch)
    model.global_threshold = valid_result['global_threshold']
    test_result = evaluation(args, model, test_loader, bce, epoch)

    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.6f}, AUPRC : {:.6f}, Best F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wprecision'], test_result['wrecall']))
    print('\tTest (DENSE) F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}, IoU : {:.6f}'.format(
        test_result['df1'], test_result['dprecision'], test_result['drecall'], test_result['dIoU']))

def main(args):

    train_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'train')
    valid_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'valid')
    test_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'test')

    train_bceloss(args, train_dataset, valid_dataset, test_dataset)

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    # for Dilated CNN
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size in the dicnn')
    parser.add_argument('--output_size', default=128, type=int, help='output size in the dicnn')
    parser.add_argument('--kernel_size', default=2, type=int, help='kernel size of the conv layers')
    parser.add_argument('--n_layers', default=7, type=int, help='# of layers in the dicnn')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    
    # for WETAS Framework
    parser.add_argument('--pooling_type', default='avg', type=str, help='avg | max')
    parser.add_argument('--local_threshold', default=0.3, type=float, help='score threshold to identify anomalies')
    parser.add_argument('--granularity', default=4, type=int, help='granularity for sequential pseudo-labels')
    parser.add_argument('--beta', default=0.1, type=float, help='margin size for the alignment loss')
    parser.add_argument('--gamma', default=0.1, type=float, help='smoothing for differentiable DTW')
    
    # for Optimization
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--n_epochs', default=200, type=int, help="# of training epochs")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--gpuidx', default=0, type=int, help='gpu index')
    parser.add_argument('--patience', default=50, type=int, help='# of patience for early stopping')
    parser.add_argument('--stopping', default='f1', type=str, help='f1 | loss')
    parser.add_argument('--seed', default=0, type=int)
    
    # for Dataset
    parser.add_argument('--dataset', default='EMG', type=str, help='EMG | GHL | SMD | Subway')
    parser.add_argument('--split_size', default=500, type=int, help='split size for preprocessing the data')

    args = parser.parse_args()

    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)

    # Random seed initialization
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)

    args.data_dir = './data/' + args.dataset

    print(args)
    main(args=args)
