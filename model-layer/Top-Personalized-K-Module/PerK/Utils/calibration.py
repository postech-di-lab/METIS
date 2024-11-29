import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from Utils.utils import *
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import random
import time

class caldset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_item, trainval_dic, val_dic, num_neg):
        super(caldset, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.trainval_dic = trainval_dic
        self.val_dic = val_dic
        self.count = None
        self.val_uiy = []
        
    def negative_sampling(self):      
        sample_list = np.random.choice(list(range(self.num_item)), size = self.num_user * self.num_neg * 15 * 5)
       
        bookmark = 0
        count = 0
        for user in range(self.num_user):
            num_neg = len(self.val_dic[user]) * self.num_neg
            trainval_list = self.trainval_dic[user]

            ## positive items
            for pos_item in self.val_dic[user]:
                self.val_uiy.append((user, pos_item, 1))
                count += 1
            
            ## negative items
            neg_list = sample_list[bookmark:bookmark+num_neg]
            bookmark = bookmark+num_neg
            _, mask, _ = np.intersect1d(neg_list, trainval_list, return_indices=True) ## sample 한것과 pos가 겹치는것

            while True:
                if len(mask) == 0:
                    break
                neg_list[mask] = sample_list[bookmark:bookmark+len(mask)]
                bookmark = bookmark+len(mask)
                _, mask, _ = np.intersect1d(neg_list, trainval_list, return_indices=True)
            for neg_item in neg_list:
                self.val_uiy.append((user, neg_item, 0))
                count += 1

        self.count = count
        self.val_uiy = np.array(self.val_uiy)

    def __len__(self):
        return self.count
        
    def __getitem__(self, idx):
        return self.val_uiy[idx][0], self.val_uiy[idx][1], self.val_uiy[idx][2]

    
class caldset_user(torch.utils.data.Dataset):
    def __init__(self, user, num_item, trainval_dic, val_dic, num_neg):
        super(caldset_user, self).__init__()
        self.num_item = num_item
        self.num_neg = num_neg
        self.trainval_dic = trainval_dic
        self.val_dic = val_dic
        self.count = None
        self.val_uiy = []
        self.user = user
        
    def negative_sampling(self):      
        sample_list = np.random.choice(list(range(self.num_item)), size = self.num_neg * len(self.val_dic) * 5)
       
        bookmark = 0
        count = 0
        
        num_neg = len(self.val_dic) * self.num_neg
        trainval_list = self.trainval_dic

        ## positive items
        for pos_item in self.val_dic:
            self.val_uiy.append((self.user, pos_item, 1))
            count += 1

        ## negative items
        neg_list = sample_list[bookmark:bookmark+num_neg]
        bookmark = bookmark+num_neg
        _, mask, _ = np.intersect1d(neg_list, trainval_list, return_indices=True) ## sample 한것과 pos가 겹치는것

        while True:
            if len(mask) == 0:
                break
            neg_list[mask] = sample_list[bookmark:bookmark+len(mask)]
            bookmark = bookmark+len(mask)
            _, mask, _ = np.intersect1d(neg_list, trainval_list, return_indices=True)
        
        ## triplet
        for neg_item in neg_list:
            self.val_uiy.append((self.user, neg_item, 0))
            count += 1
                
        self.count = count
        self.val_uiy = np.array(self.val_uiy)

    def __len__(self):
        return self.count
        
    def __getitem__(self, idx):
        return self.val_uiy[idx][0], self.val_uiy[idx][1], self.val_uiy[idx][2]

    
# https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve로 교체?
def ECELoss(scores, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.mul(torch.ones_like(scores), 0.5)).type(torch.IntTensor)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.501: # for binary classification
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.cpu().data


class Platt(nn.Module):
    def __init__(self, model, model_name, gpu):
        super(Platt, self).__init__()
        self.gpu = gpu
        
        self.model = model.cuda(self.gpu)
        self.model_name = model_name
        
        self.a = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.b = nn.Parameter(torch.ones(1) * 0).cuda(gpu)
        self.sg = nn.Sigmoid()
        
    def transform_score(self, scores):
        return self.sg(torch.mul(scores, self.a) + self.b)

    def forward(self, input):
        logits = self.model.forward_pair(input)
        return self.transform_score(logits)

    def fit_params(self, cal_loader, verbose=True):
        nll_criterion = nn.BCELoss() #.cuda(self.gpu)

        # 1) collect all the scores and labels for the validation set
        t0 = time.time()
        scores_list = []
        labels_list = []
        with torch.no_grad():
            for u, i, y in cal_loader: # validation set
                u, i, y = u.cuda(self.gpu), i.cuda(self.gpu), y.cuda(self.gpu)
                logits = self.model.forward_pair(u, i).view(-1)
                scores_list.append(logits.cpu())
                labels_list.append(y.cpu())
                del logits, u, i, y

            scores = torch.cat(scores_list) #.cuda(self.gpu)
            labels = torch.cat(labels_list).type(torch.FloatTensor) #.cuda(self.gpu)
        if verbose:
            print(time.time() - t0)
        
        # 2) Calculate NLL and ECE before calibration
        t0 = time.time()
        scores_uncal = scores.sigmoid() #.cuda(self.gpu)
        before_calibration_nll = nll_criterion(scores_uncal, labels).item()
        before_calibration_ece = ECELoss(scores_uncal, labels).item()
        if verbose:
            print('Before calibration - NLL: %.4f, ECE: %.4f' % (before_calibration_nll, before_calibration_ece))
        del scores_uncal
        torch.cuda.empty_cache()
        if verbose:
            print(time.time() - t0)

        ## https://developer.nvidia.com/blog/scikit-learn-tutorial-beginners-guide-to-gpu-accelerating-ml-pipelines/
        # 3) optimize the temperature w.r.t. NLL
        t0 = time.time()
        clf = LogisticRegression(random_state=random.randint(1, 5000000), solver='lbfgs')
        clf.fit(scores.numpy().reshape(-1, 1), labels.numpy().reshape(-1))
        self.a.data = torch.FloatTensor(clf.coef_[0]) #.cuda(self.gpu)
        self.b.data = torch.FloatTensor(clf.intercept_) #.cuda(self.gpu)   
        if verbose:
            print("fit:", time.time() - t0)

        # 4) Calculate NLL and ECE after calibration
        t0 = time.time()
        scores_cal = self.transform_score(scores) #.cuda(self.gpu)
        after_calibration_nll = nll_criterion(scores_cal, labels).item()
        after_calibration_ece = ECELoss(scores_cal, labels).item()
        if verbose:
            print('Optimal a,b: %.3f, %.3f' % (self.a.item(), self.b.item()))
            print('After calibration - NLL: %.4f, ECE: %.4f' % (after_calibration_nll, after_calibration_ece))
            print(time.time() - t0)
    
    def evaluation(self, test_loader):      
        # 1) Before calibration ============================================
        scores_list = []
        labels_list = []
        with torch.no_grad():
            for u, i, y in test_loader: # validation set
                u, i, y = u.cuda(self.gpu), i.cuda(self.gpu), y.cuda(self.gpu)
                logits = self.model.forward_pair(u, i).view(-1)
                scores_list.append(logits.cpu())
                labels_list.append(y.cpu())
                del logits, u, i, y

            scores = torch.cat(scores_list) #.cuda(self.gpu)
            labels = torch.cat(labels_list) #.cuda(self.gpu)
        
        nll_criterion = nn.BCELoss().cuda(self.gpu)
        before_calibration_nll = nll_criterion(scores.sigmoid().cuda(self.gpu), labels.type(torch.FloatTensor).cuda(self.gpu)).item()
        before_calibration_ece = ECELoss(scores.sigmoid().cuda(self.gpu), labels.type(torch.FloatTensor).cuda(self.gpu), gpu=self.gpu).item()
        print('Before calibration - NLL: %.4f, ECE: %.4f' % (before_calibration_nll, before_calibration_ece))
        
        # 2) After calibration  ============================================
        scores_cal = self.transform_score(scores.view(-1,1).cuda(self.gpu)).view(-1)
        
        after_calibration_nll = nll_criterion(scores_cal.cuda(self.gpu), labels.type(torch.FloatTensor).cuda(self.gpu)).item()
        after_calibration_ece = ECELoss(scores_cal.cuda(self.gpu), labels.type(torch.FloatTensor).cuda(self.gpu), gpu=self.gpu).item()
        print('After calibration - NLL: %.4f, ECE: %.4f' % (after_calibration_nll, after_calibration_ece))
        
        return scores, scores_cal, labels

    def ScoreDist(self, save=False):
        score_p = torch.FloatTensor(np.arange(-7, 8, 0.1)) #.cuda(self.gpu)
        score_q = self.transform_score(score_p.view(-1, 1)).view(-1).detach().cpu().numpy()

        fig = plt.figure(figsize=(4,4))
        ax = fig.subplots()
        ax.plot(score_p, score_q, color='blue') #.detach().cpu()

        ax.set_ylabel('Probability $p$', fontsize=20, color = "black")
        ax.set_xlabel('Ranking Score $s$', fontsize=20, color = "black")

        ax.xaxis.set_tick_params(labelsize=20)
        ax.set_xticks([-5,0,5])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_title("Platt", fontsize=20)

        if save:
            fig.savefig('figure/FS_Platt.png', bbox_inches="tight")