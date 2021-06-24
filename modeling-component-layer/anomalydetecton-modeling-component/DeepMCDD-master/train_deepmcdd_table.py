import os, argparse
import torch
import numpy as np

import models
from dataloader_table import get_table_data
from utils import compute_confscores, compute_metrics, print_ood_results, print_ood_results_total

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='gas | shuttle | drive | mnist')
parser.add_argument('--net_type', required=True, help='mlp')
parser.add_argument('--datadir', default='./table_data/', help='path to dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--oodclass_idx', type=int, default=0, help='index of the OOD class')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for data loader')
parser.add_argument('--latent_size', type=int, default=128, help='dimension size for latent representation') 
parser.add_argument('--num_layers', type=int, default=3, help='the number of hidden layers in MLP')
parser.add_argument('--num_folds', type=int, default=5, help='the number of cross-validation folds')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate of Adam optimizer')
parser.add_argument('--reg_lambda', type=float, default=1.0, help='regularization coefficient')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()
print(args)

def main():
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)

    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)

    best_idacc_list, best_oodacc_list = [], []
    for fold_idx in range(args.num_folds):
        
        train_loader, test_id_loader, test_ood_loader = get_table_data(args.batch_size, args.datadir, args.dataset, args.oodclass_idx, fold_idx)

        if args.dataset == 'gas':
            num_classes, num_features = 6, 128
        elif args.dataset == 'drive':
            num_classes, num_features = 11, 48
        elif args.dataset == 'shuttle':
            num_classes, num_features = 7, 9
        elif args.dataset == 'mnist':
            num_classes, num_features = 10, 784
        
        model = models.MLP_DeepMCDD(num_features, args.num_layers*[args.latent_size], num_classes=num_classes-1)
        model.cuda()

        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
   
        idacc_list, oodacc_list = [], []
        total_step = len(train_loader)
        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0.0
             
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.cuda(), labels.cuda()
                dists = model(data) 
                scores = - dists + model.alphas

                label_mask = torch.zeros(labels.size(0), model.num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)

                pull_loss = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
                push_loss = ce_loss(scores, labels)
                loss = args.reg_lambda * pull_loss + push_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                # (1) evaluate ID classification
                correct, total = 0, 0
                for data, labels in test_id_loader:
                    data, labels = data.cuda(), labels.cuda()
                    scores = - model(data) + model.alphas
                    _, predicted = torch.max(scores, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                idacc_list.append(100 * correct / total)
                
                # (2) evaluate OOD detection
                compute_confscores(model, test_id_loader, outdir, True)
                compute_confscores(model, test_ood_loader, outdir, False)
                oodacc_list.append(compute_metrics(outdir))

        best_idacc = max(idacc_list)
        best_oodacc = oodacc_list[idacc_list.index(best_idacc)]
        
        print('== {fidx:1d}-th fold results =='.format(fidx=fold_idx+1))
        print('The best ID accuracy on "{idset:s}" test samples : {val:6.2f}'.format(idset=args.dataset, val=best_idacc))
        print('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset=args.dataset+'_'+str(args.oodclass_idx)))
        print_ood_results(best_oodacc)

        best_idacc_list.append(best_idacc)
        best_oodacc_list.append(best_oodacc)

    print('== Final results ==')
    print('The best ID accuracy on "{idset:s}" test samples : {mean:6.2f} ({std:6.3f})'.format(idset=args.dataset, mean=np.mean(best_idacc_list), std=np.std(best_idacc_list)))
    print('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset='class_'+str(args.oodclass_idx)))
    print_ood_results_total(best_oodacc_list)
 

if __name__ == '__main__':
    main()
