import torch


def get_recall(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    
    n_hits = len(hits)
    recall = float(n_hits)
    return recall


def get_mrr(indices, targets):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data
    return mrr.item()


def evaluate(dist, target, K):
    recalls = []
    mrrs = []

    for k_ in K:
        _, indices = torch.topk(dist, k_, dim=-1, largest=False)
        
        recall = get_recall(indices, target)
        mrr = get_mrr(indices, target)
        recalls.append(recall)
        mrrs.append(mrr)
    return recalls, mrrs
