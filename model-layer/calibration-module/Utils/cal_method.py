import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from Utils.utils import *
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import random
from scipy.special import gamma

class Gaussian(nn.Module):
    def __init__(self, model, model_name, gpu, i_pop, smin, smax, clip):
        super(Gaussian, self).__init__()
        self.model = model
        self.model_name = model_name
        self.a = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.b = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.c = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.sg = nn.Sigmoid()
        self.gpu = gpu
        self.smin = smin
        self.smax = smax
        self.clip = clip
        self.propensities = torch.pow(i_pop / torch.max(i_pop), 0.5).cuda(gpu)

    def transform_score(self, scores):
        a = self.a.unsqueeze(1).expand(scores.size(0), scores.size(1))
        b = self.b.unsqueeze(1).expand(scores.size(0), scores.size(1))
        c = self.c.unsqueeze(1).expand(scores.size(0), scores.size(1))

        return self.sg(a*torch.pow(scores, 2) + b*scores + c) # 

    def forward(self, input):
        logits = self.model.forward_pair(input)
        return self.transform_score(logits)

    def fit_params(self, cal_loader, mode='unbiased', const=False, verbose=False, propensity_lg = None, propensity_nb = None):
        self.cuda(self.gpu)
        nll_criterion = nn.BCELoss().cuda(self.gpu)

        # 1) collect all the scores and labels for the validation set
        scores_list = []
        labels_list = []
        weights_list = []
        with torch.no_grad():
            for u, i, y in cal_loader: # biased validation label
                u, i, y = u.type(torch.LongTensor).cuda(self.gpu), i.type(torch.LongTensor).cuda(self.gpu), y.type(torch.FloatTensor).cuda(self.gpu)
                logits = self.model.forward_pair(u, i)

                # popularity
                propensities = self.propensities[i]
                propensities = torch.max(propensities, torch.ones_like(propensities) * self.clip) # propensity clipping

                # weights for pos & neg
                indicator = y.gt(torch.FloatTensor([0.5]).cuda(self.gpu)) # pos or neg -> different weight scheme
                weights_pos = (torch.ones_like(y).cuda(self.gpu) / propensities) * indicator
                weights_neg = torch.ones_like(y).cuda(self.gpu) * ~indicator

                weights_list.append(weights_pos + weights_neg)
                scores_list.append(logits)
                labels_list.append(y.view(-1,1))
                
            scores = torch.cat(scores_list).cuda(self.gpu)
            labels = torch.cat(labels_list).cuda(self.gpu)
            weights = torch.cat(weights_list)

        # 2) Calculate NLL and ECE before calibration
        before_calibration_nll = nll_criterion(scores.sigmoid(), labels).item()
        before_calibration_ece = ECELoss(scores.sigmoid(), labels, gpu=self.gpu).item()
        before_calibration_mce = MCELoss(scores.sigmoid(), labels, gpu=self.gpu).item()

        print('Before calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (before_calibration_nll, before_calibration_ece, before_calibration_mce))

        # 3) optimize the temperature w.r.t. NLL
        X = torch.cat((torch.pow(scores, 2), scores), 1).cpu().numpy()
        y = labels.cpu().numpy().reshape(-1)
        if const == False:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=random.randint(1, 5000000), solver='lbfgs')
            if mode == 'unbiased':
                clf.fit(X, y, sample_weight=weights.cpu().numpy())
            else:
                clf.fit(X, y)
        elif const == True:
            from scipy.optimize import LinearConstraint
            lb = np.array([0, 0])
            ub = np.array([np.inf, np.inf])
            A = np.zeros((2, X.shape[1] + 1))
            A[0, :2] = np.array([2*self.smin, 1])
            A[1, :2] = np.array([2*self.smax, 1])
            cons = LinearConstraint(A, lb, ub)

            from scipy.optimize import Bounds
            lb = np.r_[-np.inf, -np.inf, -np.inf]
            ub = np.r_[np.inf, np.inf, np.inf]
            bounds = Bounds(lb, ub)

            from clogistic import LogisticRegression
            clf = LogisticRegression(solver="ecos")
            if mode == 'unbiased':
                clf.fit(X, y, bounds=bounds, constraints=cons, sample_weight=weights.cpu().numpy()) #, penalty="elasticnet", l1_ratio=0.5
            else:
                clf.fit(X, y, bounds=bounds, constraints=cons)
        self.a.data = torch.FloatTensor([clf.coef_[0][0]]).cuda(self.gpu)
        self.b.data = torch.FloatTensor([clf.coef_[0][1]]).cuda(self.gpu)
        self.c.data = torch.FloatTensor(clf.intercept_).cuda(self.gpu)

        # 4) Calculate NLL and ECE after calibration
        scores_cal = self.transform_score(scores)
        after_calibration_nll = nll_criterion(scores_cal, labels).item()
        after_calibration_ece = ECELoss(scores_cal, labels, gpu=self.gpu).item()
        after_calibration_mce = MCELoss(scores_cal, labels, gpu=self.gpu).item()

        print('Optimal a,b,c: %.3f, %.3f, %.3f' % (self.a.item(), self.b.item(), self.c.item()))
        print('After calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (after_calibration_nll, after_calibration_ece, after_calibration_mce))

        return scores, labels
    
    def evaluation(self, testset, verbose=False):
        test_u = torch.LongTensor(testset[:, 0]).cuda(self.gpu)
        test_i = torch.LongTensor(testset[:, 1]).cuda(self.gpu)
        test_y = torch.LongTensor(testset[:, 2]).cuda(self.gpu)
        test_y_bi = torch.ones_like(test_y) * (test_y > 3.5) # relevance
        
        # 1) Before calibration ============================================
        self.model.eval()
        with torch.no_grad():
            scores = self.model.forward_pair(test_u, test_i).view(-1) #[idx_no3]
        
        nll_criterion = nn.BCELoss().cuda(self.gpu)
        before_calibration_nll = nll_criterion(scores.sigmoid(), test_y_bi.type(torch.FloatTensor).cuda(self.gpu)).item()
        before_calibration_ece = ECELoss(scores.sigmoid(), test_y_bi, gpu=self.gpu).item()
        before_calibration_mce = MCELoss(scores.sigmoid(), test_y_bi, gpu=self.gpu).item()

        print('Before calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (before_calibration_nll, before_calibration_ece, before_calibration_mce))
        
        # 2) After calibration  ============================================
        scores_cal = self.transform_score(scores.view(-1,1)).view(-1)
        
        after_calibration_nll = nll_criterion(scores_cal, test_y_bi.type(torch.FloatTensor).cuda(self.gpu)).item()
        after_calibration_ece = ECELoss(scores_cal, test_y_bi, gpu=self.gpu).item()
        after_calibration_mce = MCELoss(scores_cal, test_y_bi, gpu=self.gpu).item()

        print('After calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (after_calibration_nll, after_calibration_ece, after_calibration_mce))
        
        return scores, scores_cal, test_y_bi

class Gamma(nn.Module):
    def __init__(self, model, model_name, smin, smax, shift, gpu, i_pop, clip):
        super(Gamma, self).__init__()
        self.model = model
        self.model_name = model_name
        self.shift = shift

        self.smin_orig = smin
        self.smin = self.smin_orig-self.shift
        self.smax = smax
        self.clip = clip

        self.sg = nn.Sigmoid()
        self.gpu = gpu
        self.propensities = torch.pow(i_pop / torch.max(i_pop), 0.5).cuda(gpu)

        self.a = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.b = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.c = nn.Parameter(torch.ones(1) * 1).cuda(gpu)
        self.propensities = torch.pow(i_pop / torch.max(i_pop), 0.5).cuda(gpu)

    def transform_score(self, scores):
        a = self.a.unsqueeze(1).expand(scores.size(0), scores.size(1))
        b = self.b.unsqueeze(1).expand(scores.size(0), scores.size(1))
        c = self.c.unsqueeze(1).expand(scores.size(0), scores.size(1))

        return self.sg(a*torch.log(scores-self.smin) + b*(scores-self.smin) + c)

    def forward(self, input):
        logits = self.model.forward_pair(input)
        return self.transform_score(logits)

    def fit_params(self, cal_loader, mode='unbiased', const=False, verbose=False, propensity_lg = None, propensity_nb = None):
        self.cuda(self.gpu)
        nll_criterion = nn.BCELoss().cuda(self.gpu)

        # 1) collect all the scores and labels for the validation set
        scores_list = []
        labels_list = []
        weights_list = []
        with torch.no_grad():
            for u, i, y in cal_loader: # biased validation label
                u, i, y = u.type(torch.LongTensor).cuda(self.gpu), i.type(torch.LongTensor).cuda(self.gpu), y.type(torch.FloatTensor).cuda(self.gpu)
                logits = self.model.forward_pair(u, i)

                # popularity
                propensities = self.propensities[i]
                propensities = torch.max(propensities, torch.ones_like(propensities) * self.clip) # propensity clipping

                # weights for pos & neg
                indicator = y.gt(torch.FloatTensor([0.5]).cuda(self.gpu)) # pos or neg -> different weight scheme
                weights_pos = (torch.ones_like(y).cuda(self.gpu) / propensities) * indicator
                weights_neg = torch.ones_like(y).cuda(self.gpu) * ~indicator

                weights_list.append(weights_pos + weights_neg)
                scores_list.append(logits)
                labels_list.append(y.view(-1,1))
                
            scores = torch.cat(scores_list).cuda(self.gpu)
            labels = torch.cat(labels_list).cuda(self.gpu)
            weights = torch.cat(weights_list)

        # 2) Calculate NLL and ECE before calibration
        before_calibration_nll = nll_criterion(scores.sigmoid(), labels).item()
        before_calibration_ece = ECELoss(scores.sigmoid(), labels, gpu=self.gpu).item()
        before_calibration_mce = MCELoss(scores.sigmoid(), labels, gpu=self.gpu).item()

        print('Before calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (before_calibration_nll, before_calibration_ece, before_calibration_mce))

        # 3) optimize the temperature w.r.t. NLL
        X = torch.cat((torch.log(scores-self.smin), scores-self.smin), 1).cpu().numpy()
        y = labels.cpu().numpy().reshape(-1)
        if const == False:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=random.randint(1, 5000000), solver='lbfgs')
            if mode == 'unbiased':
                clf.fit(X, y, sample_weight=weights.cpu().numpy())
            else:
                clf.fit(X, y)
        elif const == True:
            from scipy.optimize import LinearConstraint
            lb = np.array([0, 0])
            ub = np.array([np.inf, np.inf])
            A = np.zeros((2, X.shape[1] + 1))
            A[0, :2] = np.array([1/self.shift, 1])
            A[1, :2] = np.array([1/(self.smax-self.smin), 1])
            cons = LinearConstraint(A, lb, ub)

            from scipy.optimize import Bounds
            lb = np.r_[-np.inf, -np.inf, -np.inf]
            ub = np.r_[np.inf, np.inf, np.inf]
            bounds = Bounds(lb, ub)

            from clogistic import LogisticRegression
            clf = LogisticRegression(solver="ecos")
            if mode == 'unbiased':
                clf.fit(X, y, bounds=bounds, constraints=cons, sample_weight=weights.cpu().numpy()) #, penalty="elasticnet", l1_ratio=0.5
            else:
                clf.fit(X, y, bounds=bounds, constraints=cons)
        self.a.data = torch.FloatTensor([clf.coef_[0][0]]).cuda(self.gpu)
        self.b.data = torch.FloatTensor([clf.coef_[0][1]]).cuda(self.gpu)
        self.c.data = torch.FloatTensor(clf.intercept_).cuda(self.gpu)

        # 4) Calculate NLL and ECE after calibration
        scores_cal = self.transform_score(scores)
        after_calibration_nll = nll_criterion(scores_cal, labels).item()
        after_calibration_ece = ECELoss(scores_cal, labels, gpu=self.gpu).item()
        after_calibration_mce = MCELoss(scores_cal, labels, gpu=self.gpu).item()

        print('Optimal a,b,c: %.3f, %.3f, %.3f' % (self.a.item(), self.b.item(), self.c.item()))
        print('After calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (after_calibration_nll, after_calibration_ece, after_calibration_mce))

        return scores, labels
    
    def evaluation(self, testset, verbose=False):
        test_u = torch.LongTensor(testset[:, 0]).cuda(self.gpu)
        test_i = torch.LongTensor(testset[:, 1]).cuda(self.gpu)
        test_y = torch.LongTensor(testset[:, 2]).cuda(self.gpu)
        test_y_bi = torch.ones_like(test_y) * (test_y > 3.5) # relevance
        
        # 1) Before calibration ============================================
        self.model.eval()
        with torch.no_grad():
            scores = self.model.forward_pair(test_u, test_i).view(-1) #[idx_no3]
        
        nll_criterion = nn.BCELoss().cuda(self.gpu)
        before_calibration_nll = nll_criterion(scores.sigmoid(), test_y_bi.type(torch.FloatTensor).cuda(self.gpu)).item()
        before_calibration_ece = ECELoss(scores.sigmoid(), test_y_bi, gpu=self.gpu).item()
        before_calibration_mce = MCELoss(scores.sigmoid(), test_y_bi, gpu=self.gpu).item()
        if verbose:
            print('Before calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (before_calibration_nll, before_calibration_ece, before_calibration_mce))
        
        # 2) After calibration  ============================================
        scores_cal = self.transform_score(scores.view(-1,1)).view(-1)
        
        after_calibration_nll = nll_criterion(scores_cal, test_y_bi.type(torch.FloatTensor).cuda(self.gpu)).item()
        after_calibration_ece = ECELoss(scores_cal, test_y_bi, gpu=self.gpu).item()
        after_calibration_mce = MCELoss(scores_cal, test_y_bi, gpu=self.gpu).item()

        print('After calibration - NLL: %.4f, ECE: %.4f, MCE: %.4f' % (after_calibration_nll, after_calibration_ece, after_calibration_mce))
        
        return scores, scores_cal, test_y_bi



