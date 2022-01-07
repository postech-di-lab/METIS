#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import matplotlib.pyplot as plt

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        print('# Load the CIFAR dataset')
        data_dir = '../Dataset/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # Sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            print('# Load the MNIST dataset')
            data_dir = '../Dataset/mnist/'
        else:
            print('# Load the Fashion MNIST dataset')
            data_dir = '../Dataset/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def print_arguments(args):
    """
    Print the main arguments
    """
    print(f'\n# Base parameters')
    print(f'## Model     : {args.model}')
    print(f'## Optimizer : {args.optimizer}')
    print(f'## Learning rate  : {args.lr}')
    print(f'## Global epochs  : {args.epochs}\n')

    print(f'# Federated parameters')
    if args.iid:
        print(f'## IID option:  IID')
    else:
        print(f'## IID option:  Non-IID')
    print(f'## Fraction of users  : {args.frac}')
    print(f'## Local batch size   : {args.local_bs}')
    print(f'## Local epochs       : {args.local_ep}\n')
    

def save_loss_curve(args, epoch_loss):
    """
    Save the loss curve
    """
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('./Figure/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))
    print(f'Save the loss curve at ./Figure/nn_{args.dataset}_{args.model}_{args.epochs}.png')