import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from dataset import MnistDataset
from model import MLP
from utils import set_seed


def main(seed):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    # print(f"Using device: {device}","seed:",seed)
    # Hyperparameters
    batch_size = 256
    learning_rate = 0.01
    num_epochs = 5

    raw_train_dataset = MnistDataset(train=True)
    raw_test_dataset = MnistDataset(train=False)
    # print(len(raw_test_dataset),len(raw_train_dataset))
    # exit()
    raw_size = raw_train_dataset[0][0].shape[1] * raw_train_dataset[0][0].shape[2]

    raw_train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=True)
    raw_test_loader = DataLoader(raw_test_dataset, batch_size=batch_size, shuffle=False)

    comp_train_dataset = MnistDataset(train=True, compress=True)
    comp_test_dataset = MnistDataset(train=False, compress=True)
    comp_size = comp_train_dataset[0][0].shape[1] * comp_train_dataset[0][0].shape[2]

    comp_train_loader = DataLoader(comp_train_dataset, batch_size=batch_size, shuffle=True)
    comp_test_loader = DataLoader(comp_test_dataset, batch_size=batch_size, shuffle=False)

    base_model = MLP(raw_size).to(device)
    approx_model = MLP(comp_size).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    base_optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    approx_optimizer = optim.Adam(approx_model.parameters(), lr=learning_rate)

    # Training the model
    # print("Training started...")
    base_model.train()
    approx_model.train()

    base_training_times = []
    approx_training_times = []

    for epoch in range(num_epochs):
        # Base full epoch
        start_time1 = time.time()
        for images, labels in raw_train_loader:
            base_optimizer.zero_grad()
            outputs = base_model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            base_optimizer.step()
        end_time1 = time.time()
        base_training_time = end_time1 - start_time1
        base_training_times.append(base_training_time)

        # Approx epoch with time limit
        start_time2 = time.time()
        time_limit = base_training_time * 0.8
        for images, labels in comp_train_loader:
            approx_optimizer.zero_grad()
            outputs = approx_model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            approx_optimizer.step()
            if time.time() > start_time2 + time_limit:
                break
        end_time2 = time.time()
        approx_training_times.append(end_time2 - start_time2)

    # Testing the model
    base_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in raw_test_loader:
            outputs = base_model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    base_acc = 100 * correct / total

    approx_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in comp_test_loader:
            outputs = approx_model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    approx_acc = 100 * correct / total

    return (sum(base_training_times), sum(approx_training_times)), min([approx_acc / base_acc, 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('-s', type=int, help='seed')
    args = parser.parse_args()
    seed = args.s

    main(seed)
