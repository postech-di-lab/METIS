#
# This file is implemented based on the author code of
#    Lee et al., "A simple unified framework for detecting out-of-distribution samples and adversarial attacks", in NeurIPS 2018.
#

import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

def get_svhn(batch_size, train_TF, test_TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=train_TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=test_TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar10(batch_size, train_TF, test_TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=train_TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=test_TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, train_TF, test_TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=train_TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=test_TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_id_image_data(data_type, batch_size, train_TF, test_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = get_cifar10(batch_size=batch_size, train_TF=train_TF, test_TF=test_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = get_cifar100(batch_size=batch_size, train_TF=train_TF, test_TF=test_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = get_svhn(batch_size=batch_size, train_TF=train_TF, test_TF=test_TF, data_root=dataroot, num_workers=1)

    return train_loader, test_loader

def get_ood_image_data(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = get_cifar10(batch_size=batch_size, train_TF=input_TF, test_TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = get_svhn(batch_size=batch_size, train_TF=input_TF, test_TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = get_cifar100(batch_size=batch_size, train_TF=input_TF, test_TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet_crop':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_crop'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_crop':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_crop'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return test_loader


