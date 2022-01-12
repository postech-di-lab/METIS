#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import sys
import pandas as pd
import matplotlib as mpl
import random

import torch
from tensorboardX import SummaryWriter

from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, get_initial_model
from utils import get_dataset, average_weights, print_arguments, save_train_loss_and_acc, save_figures
from train import train_federated


#################################################################################################################
# Define Run function (Federated)
#################################################################################################################
def run():
    start_time = time.time()
    print("="*20,"Settings...","="*20)
    # Define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    print_arguments(args)

    # Set training device
    if args.gpu != -1:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(device) # change allocation of current GPU
        print(f"# Training device: {device, torch.cuda.get_device_name()}")
    else:
        print('cpu')
        device = torch.device("cpu")
        print(f"# Training device: {device}")
    print("="*20,"...Settings","="*20); print() 
    
    print("="*20,"Dataset Load...","="*20)
    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print("="*20,"...Dataset Load","="*20); print()
    
    print("="*20,"Model Initialization...","="*20)
    # Get initialized model
    global_model = get_initial_model(args, device, train_dataset)

    # Copy weights
    global_weights = global_model.state_dict()
    print("="*20,"...Model Initialization","="*20); print()
    
    print("="*20,"Model Training...","="*20)
    # Training
    global_model, train_loss, train_accuracy = train_federated(args, device, global_model, global_weights, train_dataset, logger, user_groups)

    # Testing
    test_acc, test_loss = test_inference(args, device, global_model, test_dataset)
    print('# Test on', len(test_dataset), 'samples')
    print(f'## Results after {args.epochs} global rounds of training:')
    print("### Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("### Test Accuracy: {:.2f}%".format(100*test_acc))
    
    # Save figure
    save_train_loss_and_acc(args, train_loss, train_accuracy)
    save_figures(args, train_loss, train_accuracy)
    print("="*20,"...Model Training","="*20); print()
    
    print('\n# Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    
#################################################################################################################
# Define Arguments and Main function
#################################################################################################################
if __name__ == '__main__':
    #################################################################################################################
    # Library Version
    #################################################################################################################
    print("="*20,"Library Version...","="*20)
    print(f"python version: {sys.version}")
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {mpl.__version__}")
    print(f"torch version: {torch.__version__}")
    print("="*20,"...Library Version","="*20); print()

    #################################################################################################################
    # Reproducible
    #################################################################################################################
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42) # if use mult-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    #################################################################################################################
    # Hyperparameters Setting
    #################################################################################################################
    parser = argparse.ArgumentParser()

    # Federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # Model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # Other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=-1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    
    # Run
    run()