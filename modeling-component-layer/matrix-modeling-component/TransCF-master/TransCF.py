import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from Recommender import Recommender
import evaluation
from Dataset import Dataset_TransCF
from torch.backends import cudnn
from torch.utils.data import DataLoader
import random
import fastrand

class TransCF(Recommender):
    def __init__(self, args):
        Recommender.__init__(self, args)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        
        if self.cuda_available == True:
            self.clip_max = torch.FloatTensor([1.0]).cuda(self.cuda)
        else:
            self.clip_max = torch.FloatTensor([1.0])
        
        self.test = self.getTestInstances()

    # Training
    def training(self):
        model = modeler(self.numUsers, self.numItems, self.embedding_dim, self.cuda_available, self.cuda)
        if self.cuda_available == True: 
            model = model.cuda(self.cuda)
        
        criterion = torch.nn.MarginRankingLoss(margin=self.margin)
        optimizer = optim.SGD(model.parameters(), lr = self.lRate)
        
        # Initial performance
        model.eval()
        topHits, topNdcgs, topMrrs = self.evalScore(model)
        model.train()
        
        print("[%s] [Initial %s] %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f"%(self.recommender, self.currentTime(), topHits[1], topHits[5], topHits[10], topHits[20], topHits[50], topNdcgs[1], topNdcgs[5], topNdcgs[10], topNdcgs[20], topNdcgs[50], topMrrs[1], topMrrs[5], topMrrs[10], topMrrs[20], topMrrs[50]))
        

        bestHR = 0
        bestNDCG = 0
        early_stop_metric = []
        for epoch in range(self.numEpoch):
            totalLoss = 0
            # Reading Data
            totalData = self.getTrainInstances()
            train_by_dataloader = Dataset_TransCF(totalData)
            train_loader = DataLoader(dataset=train_by_dataloader, batch_size=self.batch_size, shuffle=True)
            for batch_idx, batch in enumerate(train_loader):
                u = Variable(batch['u'])
                i = Variable(batch['i'])
                j = Variable(batch['j'])
                
                i_viewed_u_idx, i_viewed_u_offset, u_viewed_i_idx, u_viewed_i_offset = self.getNeighbors(u, i)
                j_viewed_u_idx, j_viewed_u_offset, u_viewed_j_idx, u_viewed_j_offset = self.getNeighbors(u, j)

                if self.cuda_available == True:
                    u = u.cuda(self.cuda); i = i.cuda(self.cuda); j = j.cuda(self.cuda)
                    
                optimizer.zero_grad()
                
                # Observed (positive) interaction
                pos, reg = model(u, i, u_viewed_i_idx, u_viewed_i_offset, i_viewed_u_idx, i_viewed_u_offset)
                # Unobserved (negative) interaction
                neg, _ = model(u, j, u_viewed_j_idx, u_viewed_j_offset, j_viewed_u_idx, j_viewed_u_offset)

                if self.cuda_available == True:
                    loss = criterion(pos, neg, Variable(torch.FloatTensor([-1])).cuda(self.cuda))
                else:
                    loss = criterion(pos, neg, Variable(torch.FloatTensor([-1])))
                
                for elem, regx in zip([self.reg1, self.reg2], reg):
                    loss += elem * regx
                
                loss.backward()
                optimizer.step()

                totalLoss += loss.data[0]
            
            # Unit-norm regularization
            model.userEmbed.weight.data.div_(torch.max(torch.norm(model.userEmbed.weight.data, 2, 1, True), self.clip_max).expand_as(model.userEmbed.weight.data))
            model.itemEmbed.weight.data.div_(torch.max(torch.norm(model.itemEmbed.weight.data, 2, 1, True), self.clip_max).expand_as(model.itemEmbed.weight.data))
            
            # Evaluate the performance every three iterations (for running time issue)
            if epoch % 3 == 0:
                model.eval()
                topHits, topNdcgs, topMrrs = self.evalScore(model)
                model.train()
            
            if self.is_converged(model, epoch, totalLoss, topHits, topNdcgs, topMrrs):
                return
            
        self.printFinalResult()
    
    
    
    
class modeler(nn.Module):
    def __init__(self, numUsers, numItems, embedding_dim, cuda_available, gpunum):
        super(modeler, self).__init__()
        self.userEmbed = nn.EmbeddingBag(numUsers, embedding_dim, mode='mean')
        self.itemEmbed = nn.EmbeddingBag(numItems, embedding_dim, mode='mean')
        self.cuda_available = cuda_available
        self.init_weights()
        self.gpunum = gpunum
        
    def init_weights(self):
        nn.init.normal(self.userEmbed.weight.data, mean=0.0, std=0.01)
        nn.init.normal(self.itemEmbed.weight.data, mean=0.0, std=0.01)

    def forward(self, u, i, u_viewed_i_idx, u_viewed_i_offset, i_viewed_u_idx, i_viewed_u_offset):
        userIdx = Variable(torch.LongTensor(range(0,len(u))))
        itemIdx = Variable(torch.LongTensor(range(0,len(i))))
        if self.cuda_available == True:
            userIdx = userIdx.cuda(self.gpunum)
            itemIdx = itemIdx.cuda(self.gpunum)
            
        userEmbeds = self.userEmbed(u, userIdx)
        itemEmbeds = self.itemEmbed(i, itemIdx)
        
        # Get neighborhood embeddings
        userNeighborEmbeds = self.itemEmbed(i_viewed_u_idx, i_viewed_u_offset)
        itemNeighborEmbeds = self.userEmbed(u_viewed_i_idx, u_viewed_i_offset)
        
        # Get r_{ui}
        rel = userNeighborEmbeds * itemNeighborEmbeds
        
        # Distance Regularizer
        tmp = (userEmbeds + rel - itemEmbeds)**2
        reg1 = tmp.sum()

        # Neighborhood Regularizer
        reg2 = ((userEmbeds - userNeighborEmbeds)**2).sum() + ((itemEmbeds - itemNeighborEmbeds)**2).sum()
        out = torch.sum(tmp,1)
        
        # Gather regularizers
        reg = [reg1, reg2]
        
        return out, reg
