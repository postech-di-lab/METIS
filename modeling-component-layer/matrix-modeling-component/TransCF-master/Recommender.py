import time
import pandas as pd
import math
from Dataset import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import heapq
import evaluation
import fastrand

class Recommender(object):
    def __init__(self, args):
        self.cuda_available = torch.cuda.is_available()
        self.recommender = args.recommender
        self.numEpoch = args.numEpoch
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim
        self.lRate = args.lRate
        self.topK = eval(args.topK)
        self.reg1 = args.reg1
        self.reg2 = args.reg2
        self.num_negatives = args.num_negatives
        self.dataset = args.dataset
        self.margin = args.margin
        self.rand_seed = args.rand_seed
        np.random.seed(self.rand_seed)
        self.mode = args.mode
        self.cuda = args.cuda
        self.batchSize_test = args.batchSize_test
        self.early_stop = args.early_stop
        
        self.totalFilename = 'data/'+self.dataset+'/ratings.dat'
        self.trainFilename = 'data/'+self.dataset+'/LOOTrain.dat'
        self.valFilename = 'data/'+self.dataset+'/LOOVal.dat'
        self.testFilename = 'data/'+self.dataset+'/LOOTest.dat'
        self.negativesFilename = 'data/'+self.dataset+'/LOONegatives.dat'
        
        dataset = Dataset(self.totalFilename, self.trainFilename, self.valFilename, self.testFilename, self.negativesFilename)
        self.trainRatings, self.valRatings, self.testRatings, self.negatives, self.numUsers, self.numItems, self.userCache, self.itemCache = dataset.trainMatrix, dataset.valRatings, dataset.testRatings, dataset.negatives, dataset.numUsers, dataset.numItems, dataset.userCache, dataset.itemCache
        
        
        self.train = dataset.train
        self.totalTrainUsers, self.totalTrainItems = dataset.totalTrainUsers, dataset.totalTrainItems                
            
        # Evaluation
        self.bestHR1 = 0; self.bestNDCG1 = 0; self.bestMRR1 = 0; self.bestHR5 = 0; self.bestNDCG5 = 0; self.bestMRR5 = 0; self.bestHR10 = 0; self.bestNDCG10 = 0; self.bestMRR10 = 0; self.bestHR20 = 0; self.bestNDCG20 = 0; self.bestMRR20 = 0; self.bestHR50 = 0; self.bestNDCG50 = 0; self.bestMRR50 = 0; 
        self.early_stop_metric = []
        
    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    
    def is_converged(self, model, epoch, totalLoss, topHits, topNdcgs, topMrrs):
        
        HR1 = topHits[1]
        NDCG1 = topNdcgs[1]
        MRR1 = topMrrs[1]
        if HR1 > self.bestHR1:
            self.bestHR1 = HR1
        if NDCG1 > self.bestNDCG1:
            self.bestNDCG1 = NDCG1
        if MRR1 > self.bestMRR1:
            self.bestMRR1 = MRR1
        
        HR5 = topHits[5]
        NDCG5 = topNdcgs[5]
        MRR5 = topMrrs[5]
        if HR5 > self.bestHR5:
            self.bestHR5 = HR5
        if NDCG5 > self.bestNDCG5:
            self.bestNDCG5 = NDCG5
        if MRR5 > self.bestMRR5:
            self.bestMRR5 = MRR5
        
        HR10 = topHits[10]
        NDCG10 = topNdcgs[10]
        MRR10 = topMrrs[10]
        if HR10 > self.bestHR10:
            self.bestHR10 = HR10            
        if NDCG10 > self.bestNDCG10:
            self.bestNDCG10 = NDCG10
        if MRR10 > self.bestMRR10:
            self.bestMRR10 = MRR10
        
        HR20 = topHits[20]
        NDCG20 = topNdcgs[20]
        MRR20 = topMrrs[20]
        if HR20 > self.bestHR20:
            self.bestHR20 = HR20
        if NDCG20 > self.bestNDCG20:
            self.bestNDCG20 = NDCG20
        if MRR20 > self.bestMRR20:
            self.bestMRR20 = MRR20
        
        HR50 = topHits[50]
        NDCG50 = topNdcgs[50]
        MRR50 = topMrrs[50]
        if HR50 > self.bestHR50:
            self.bestHR50 = HR50
        if NDCG50 > self.bestNDCG50:
            self.bestNDCG50 = NDCG50
        if MRR50 > self.bestMRR50:
            self.bestMRR50 = MRR50

        if epoch % 10 == 0:
            print("[%s] [iter=%d %s] Loss: %.2f, margin: %.3f | %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f"%(self.recommender, epoch+1, self.currentTime(), totalLoss, self.margin, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))

        self.early_stop_metric.append(self.bestHR10)
        if self.mode == 'Val' and epoch > self.early_stop and self.bestHR10 == self.early_stop_metric[epoch-self.early_stop]:
            print("[%s] [Final (Early Converged)] %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f" %(self.recommender, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))
            return True
        
    def printFinalResult(self):
        print("[%s] [Final] %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f" %(self.recommender, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))

    def evalScore(self, model):
        topHits = dict(); topNdcgs = dict(); topMrrs = dict()
        trainItems = set(self.train.iid.unique())
        for topK in self.topK:
            hits = []; ndcgs = []; mrrs = []
            for idx in range(len(self.test.keys())):
                users = Variable(self.test[idx]['u'])
                items = Variable(self.test[idx]['i'])
                offsets = self.test[idx]['offsets']
                
                i_viewed_u_idx, i_viewed_u_offset, u_viewed_i_idx, u_viewed_i_offset = self.getNeighbors(users, items)

                map_item_score = {}

                vals, _ = model(users, items, u_viewed_i_idx, u_viewed_i_offset, i_viewed_u_idx, i_viewed_u_offset)
                vals *= -1.0
                if self.cuda_available == True:
                    items = items.cpu().data.numpy().tolist()
                    vals = vals.cpu().data.numpy().tolist()
                    #torch.cuda.empty_cache()
                else:
                    items = items.data.numpy().tolist()
                    vals = vals.data.tolist()

                for i in range(len(offsets)-1):
                    from_idx = offsets[i]
                    to_idx = offsets[i+1]
                    cur_items = items[from_idx:to_idx]
                    cur_vals = vals[from_idx:to_idx]

                    gtItem = cur_items[-1]
                    
                    map_item_score = dict(zip(cur_items, cur_vals))

                    ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
                    hr = evaluation.getHitRatio(ranklist, gtItem)
                    ndcg = evaluation.getNDCG(ranklist, gtItem)
                    mrr = evaluation.getMRR(ranklist, gtItem)

                    hits.append(hr)
                    ndcgs.append(ndcg) 
                    mrrs.append(mrr)

            hr, ndcg, mrr = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(mrrs).mean()
            topHits[topK] = hr; topNdcgs[topK] = ndcg; topMrrs[topK] = mrr
            
        return topHits, topNdcgs, topMrrs
    
    def getNeighbors(self, uids, iids):
        uid_idxvec = []
        uid_offset = []
        prev_len = 0
        if self.cuda_available == True:
            iids = iids.cpu().data.numpy().tolist()
        else:
            iids = iids.data.numpy().tolist()
        
        for iid in iids:
            users = self.itemCache[iid]
            uid_idxvec += users
            uid_offset.append(prev_len)
            prev_len += len(users)
            
            
        iid_idxvec = []
        iid_offset = []
        prev_len = 0
        if self.cuda_available == True:
            uids = uids.cpu().data.numpy().tolist()
        else:
            uids = uids.data.numpy().tolist()
            
        for uid in uids:
            items = self.userCache[uid]
            iid_idxvec += items
            iid_offset.append(prev_len)
            prev_len += len(items)
        
        if self.cuda_available == True:
            return Variable(torch.LongTensor(iid_idxvec)).cuda(self.cuda), Variable(torch.LongTensor(iid_offset)).cuda(self.cuda), Variable(torch.LongTensor(uid_idxvec)).cuda(self.cuda), Variable(torch.LongTensor(uid_offset)).cuda(self.cuda)
        else:
            return Variable(torch.LongTensor(iid_idxvec)), Variable(torch.LongTensor(iid_offset)), Variable(torch.LongTensor(uid_idxvec)), Variable(torch.LongTensor(uid_offset))
    
    def getTestInstances(self):
        trainItems = set(self.train.iid.unique())
        test=dict()
        # Make test data
        input = range(self.numUsers)
        bins = [input[i:i+self.batchSize_test] for i in range(0, len(input), self.batchSize_test)]

        for bin_idx, bin in enumerate(bins):
            userIdxs = []
            itemIdxs = []
            prevOffset = 0
            offset = [0]
            for uid in bin:
                if self.mode == 'Val':
                    rating = self.valRatings[uid]
                else:
                    rating = self.testRatings[uid]
                items = self.negatives[uid]
                items = list(trainItems.intersection(set(items)))
                u = rating[0]
                assert (uid == u)
                gtItem = rating[1]
                if gtItem not in trainItems:
                    continue
                items.append(gtItem)

                users = [u] * len(items)

                userIdxs += users
                itemIdxs += items
                offset.append(prevOffset + len(users))
                prevOffset += len(users)

            test.setdefault(bin_idx, dict())
            test[bin_idx]['offsets'] = offset
            if self.cuda_available == True:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs)).cuda(self.cuda)
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs)).cuda(self.cuda)

            else:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs))
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs))
                
        return test
    
    
    def getTrainInstances(self):
        trainItems = set(self.train.iid.unique())
        totalData = []
        for s in range(self.numUsers * self.num_negatives):
            while True:
                u = fastrand.pcg32bounded(self.numUsers)
                cu = self.userCache[u]
                if len(cu) == 0:
                    continue

                t = fastrand.pcg32bounded(len(cu))
                
                #i = list(cu)[t]
                i = cu[t]
                j = fastrand.pcg32bounded(self.numItems)

                while j in cu or j not in trainItems:
                    j = fastrand.pcg32bounded(self.numItems)
                    
                break

            totalData.append([u, i, j])
                
        totalData = np.array(totalData)
        
        return totalData
        
