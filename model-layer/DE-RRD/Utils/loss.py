import torch

def relaxed_ranking_loss(S1, S2):

	above = S1.sum(1, keepdims=True)

	below1 = S1.flip(-1).exp().cumsum(1)		
	below2 = S2.exp().sum(1, keepdims=True)		

	below = (below1 + below2).log().sum(1, keepdims=True)
	
	return -(above - below).sum()

