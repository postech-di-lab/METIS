# FedAvg: Federated Averaging

## Overview
> Recently, the data collected from mobile devices is suitable and abundant for individuals who use devices. At the same time, these data are sensitive to privacy and may be too much to learn on a mobile device. For this reason, there is a difficulty in using the existing approach as it is. To solve this problem, we propose federated learning that to learn a shared model by aggregating locally-computed updates while leaving the training data on the mobile devices. To explain the effectiveness of the methodology, experiments were conducted using simple datasets and models such as MNIST and MLP.

## Paper
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (*AISTATS, 2017*)](https://arxiv.org/abs/1602.05629)
  - **H. Brendan McMahan**, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agiiera y Arcas

## Reqirements
- Install all the packages from requirements.txt in your virtual environment. 
```
pip install -r requirements.txt
```

## Dataset
- Refer to the /Dataset/README.md

## Configuration

- **Base Parameters**

Parameter | Default | Description
--- | :---: | ---
`--dataset <str>` | 'mnist' | The name of dataset
`--model <str>` | 'mlp' | The name of model
`--gpu <int>` | -1 | To use cuda, set to a specific GPU ID. Default set to use CPU
`--epochs <int>` | 10 | The number of rounds of training
`--lr <float>` | 0.01 | The learning rate
`--verbose <int>` | 1 | 0: Activate the detailed log outputs, 1: Deactivate
`--seed <int>` | 1 | The random seed

- **Parameters for federated learning**

Parameter | Default | Description
--- | :---: | ---
`--iid <int>` | 1 | The distribution of data amongst user. 0: Set to IID, 1: Set to non-IID
`--num_users <int>` | 100 |The number of users
`--frac <float>` | 0.1 | The fraction of clients
`--local_ep <int>` | 10 | The number of local epochs
`--local_bs <int>` | 10 | The local batch size
`--unequal <int>` | 0 | Whether to use unequal data splits for non-IID setting. 0: Equal splits, 1: Unequal splits

## Usage

A. For the baseline (Standard SGD)

```
python run_baseline.py --model mlp --dataset mnist --gpu 0
```

B. For the federated learning (Proposed)

```
python run_federated.py --model mlp --dataset mnist --gpu 0
```

## Results

The experiment was conducted on the MNIST dataset.

Method | Accuracy(%)
--- | :---: 
`Baseline` | 92.68 
`FedAVG` | 91.51 

