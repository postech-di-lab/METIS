import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import pickle
from models import PERIS
from metric import cal_measures, get_each_score, get_logit, get_pis
from dataloaders.dataloader_seqrs import DataLoader_seq

torch.set_num_threads(1)

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

class Instructor:
    def __init__(self, opt):
        self.opt = opt                        

        self.data_loader = DataLoader_seq(self.opt)                        

        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        
        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()
        
    def train(self):        
        newtime = round(time.time())        
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)      
               
        best_score = -1 
        best_topHits, best_topNdcgs = None, None
        batch_loss = 0
        batch_loss_aux = 0
        c = 0 # to check early stopping
        
        for epoch in range(self.opt.num_epoch):
            st = time.time()
    
            for i, batch_data in enumerate(self.trn_loader):
                batch_data = [bd.cuda() for bd in batch_data]                                 
                optimizer.zero_grad() 
                
                if opt.model_name in ['peris']:                    
                    
                    if epoch < self.opt.warmup_epochs:
                        loss = self.model.compute_warmup_loss(batch_data)          
                    else:
                        loss, loss_IS = self.model.compute_loss(batch_data)
                        
                else:
                    loss = self.model.compute_loss(batch_data)                
                    
                loss.backward()
                
                optimizer.step()
    
                batch_loss += loss.data.item()
                
                if opt.model_name in ['peris'] and epoch>=self.opt.warmup_epochs:
                    batch_loss_aux += loss_IS.data.item()

            elapsed = time.time() - st
            evalt = time.time()
            
            with torch.no_grad():
                topHits, topNdcgs  = cal_measures(self.vld_loader, self.model, opt, 'vld')                
                
                if (topHits[10] + topNdcgs[10])/2 > best_score:
                    best_score = (topHits[10] + topNdcgs[10])/2
                    
                    best_topHits = topHits
                    best_topNdcgs = topNdcgs
                    
                    c = 0
                    
                    test_topHits, test_topNdcgs = cal_measures(
                                    self.tst_loader, self.model, opt, 'tst')
                    
                    if opt.save == True:          
                        torch.save(self.model.ebd_user.weight, 
                                   opt.save_path+'/useremb_{}_{}.pth'.format(opt.model_name, opt.dataset))
                        torch.save(self.model.ebd_item.weight, 
                                   opt.save_path+'/itememb_{}_{}.pth'.format(opt.model_name, opt.dataset))

                        logit = get_logit(self.tst_loader, self.model, opt, 'tst')
                        np.save(opt.save_path+'/tstlogit_{}_{}.npy'.format(opt.model_name, opt.dataset), logit)

                        if opt.model_name == 'peris': # Save PIS score
                            pis = get_pis(self.tst_loader, self.model, opt, 'tst')
                            np.save(opt.save_path+'/tstpis_{}_{}.npy'.format(opt.model_name, opt.dataset), pis)                          

                        # # Code for saving the model's predictions                
                        # topHits4save, topNdcgs4save = get_each_score(self.tst_loader, self.model, opt, 'tst')
                        
                        # with open('prediction/{}_{}_hit'.format(opt.model_name, opt.dataset), 'wb') as fw:
                        #     pickle.dump(topHits4save, fw)
                        # with open('prediction/{}_{}_ndcg'.format(opt.model_name, opt.dataset), 'wb') as fw:
                        #     pickle.dump(topNdcgs4save, fw)

                    
                evalt = time.time() - evalt 
            
            if opt.model_name in ['peris']:            
                print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, TRN_IS_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), batch_loss_aux/len(self.trn_loader), (topHits[10] + topNdcgs[10])/2,  test_topHits[10])))                               
            else:            
                print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), (topHits[10] + topNdcgs[10])/2,  test_topHits[10])))

            batch_loss = 0
            batch_loss_aux = 0

            c += 1
            
            if opt.model_name in ['peris']:
                if epoch < self.opt.warmup_epochs:
                    c = 0 # Don't count patient during warm-up steps
            
            if c > 5: break # Early-stopping
        
        print(('\nValid score@10 : %5.4f, HR@10 : %5.4f, NDCG@10 : %5.4f\n'% (((best_topHits[10] + best_topNdcgs[10])/2), best_topHits[10],  best_topNdcgs[10])))
        
        return test_topHits,  test_topNdcgs, best_score, best_topHits[10], best_topNdcgs[10]
            
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        print('')

    def run(self, repeats):
        results = []
        rndseed = [312596, 970456, 303598, 970542, 513897] # random seeds
        for i in range(repeats):
            print('\n💫 run: {}/{}'.format(i+1, repeats))
            
            if self.opt.model_name in ['peris']:
                print('\nWarmup up to {}-th epoch\n'.format(self.opt.warmup_epochs))
            
            random.seed(rndseed[i]); np.random.seed(rndseed[i]); torch.manual_seed(rndseed[i])            

            self.model = self.opt.model_class(self.opt).cuda()
            
            results.append(ins.train())
        
        results = np.array(results)
        
        best_vld_scores = results[:,2].mean()
        best_vld_HR = results[:,3].mean()
        best_vld_nDCG = results[:,4].mean()
        print('\nBest VLD scores (mean): {:.4}\tHR@10:\t{:.4}\tnDCG@10:\t{:.4}\n'.format(best_vld_scores, best_vld_HR, best_vld_nDCG))
        
        hrs_mean = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_mean = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
        hrs_std = np.array([list(i.values()) for i in results[:,0]]).std(0)
        ndcg_std = np.array([list(i.values()) for i in results[:,1]]).std(0)
        
        print('*TST STD\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_std.astype(str))))
        print('*NDCG means: {}\n'.format(', '.join(ndcg_std.astype(str))))
        
    
        print('*TST Performance\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_mean.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_mean.astype(str))))
        
    def _reset_params(self):
        self.model = self.opt.model_class(self.opt).cuda()
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='peris', type=str)
    parser.add_argument('--dataset', default='tools', type=str)    
    parser.add_argument('--num_run', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)    
    parser.add_argument('--batch_size', default=128, type=int)    
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--num_worker', default=4, type=int)    

    # HPs for general RS
    parser.add_argument('--margin', default=0.6, type=float)    
    parser.add_argument('--K', default=128, type=int)      
    parser.add_argument('--numneg', default=5, type=int)  

    # HPs for PERIS
    parser.add_argument('--lamb', default=0.5, type=float) # Equation 7 and 8
    parser.add_argument('--mu', default=0.3, type=float)  # Equation 7 and 8

    parser.add_argument('--binsize', default=8, type=int) # $w$ in the paper (section 3.2.2)
    parser.add_argument('--period', default=64, type=int) # It controls $T$ in the paper (section 3.2.2)
    parser.add_argument('--tau', default=0, type=float) # $\tau$ in the paper (section 3.3)
    parser.add_argument('--neg_weight', default=1.0, type=float)
    parser.add_argument('--bin_ratio', default=0.5, type=float)    

    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--aggtype', default='max', type=str, help='sum, mean, max')
    parser.add_argument('--maxhist', default=100, type=int) # The maximum # of consumed items per user
    
    opt = parser.parse_args()   

    if 'yelp' in opt.dataset: opt.neg_weight = 0.1       
    
    torch.cuda.set_device(opt.gpu)
    
    model_classes = {         
        'peris':PERIS, 
    }  
    
    dataset_path = './data/{}/rec'.format(opt.dataset)    
    opt.save_path = './trained_models/'
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path

    ins = Instructor(opt)
    
    ins.run(opt.num_run)     
