# **Text-Graph Modeling Component**

A PyTorch implementation of **Relational-Sequential Modeling Component** that processes sequential data. 
More specifically, our dataset's line contains an user id, item id, rating and timestamp. 
Sort these interaction items in chronological order and configure the data sequentially.
Using these, it learns how to predict next item for a given user history.

## Details

### Dataset

It uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset.

### Requirements

It requires the latest version of:

- pandas
- numpy
- [pytorch](https://pytorch.org)

It was tested for Python 3.7, PyTorch 1.7.0 and CUDA 11.0.

### To Run

```none
python main.py
```

### To Test

```none
python test.py
```

### Results

```none
Epoch: 28
# Val mAP: 0.1669
Epoch: 29
# Val mAP: 0.1681
## Test mAP: 0.1569
Epoch: 30
# Val mAP: 0.169
## Test mAP: 0.1601
```
