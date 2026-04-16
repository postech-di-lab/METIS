import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't worsen after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, larger_is_better = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.stop_flag = False
        self.larger_is_better = larger_is_better
        self.best_score = -torch.inf if larger_is_better else torch.inf
        self.delta = delta
        
    def __call__(self, score):
        if  self.larger_is_better:
            if score <= self.best_score + self.delta:  # Change the condition to check for worsening
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop_flag = True
            elif score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
        else:
            if score >= self.best_score + self.delta:  # Change the condition to check for worsening
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop_flag = True
            elif score < self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
