# FedAvg: Federated Avaeraging

## Overview

## Paper
- Communication-Efficient Learning of Deep Networks from Decentralized Data (*AISTATS, 2017*)
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
`--dataset <int>` | 4 | The number of concurrent threads
`-g <int>` | 16 | The size of a grid
`-d <int>` | 0 | 0 : build the GCSC file on memory, 1 : build the GCSC file on disk

- **Parameters for federated learning**

Parameter | Default | Description
--- | :---: | ---
`-k <int>` | 40 | The rank size of factor matrices
`-l <float>` | 0.01 | The regularization coefficient
`-t <int>` | 5 | The number of total iterations
`-n <int>` | 4 | The number of concurrent threads
`-g <int>` | 16 | The size of a grid
`-o <int>` | 0 | 0 : OCAM, 1 : OCAM-opt
## Usage

## Results

## Reference
