# **Text-Graph Modeling Component**

A PyTorch implementation of **Text-Graph Modeling Component** that processes data consisting of text and graph. More specifically, it loads user-item ratings in graph form and reviews in text form. Using these, it learns how to predict ratings for a given user and item.

## Details

### Dataset

It uses [Amazon 'Movies and TV'](https://nijianmo.github.io/amazon/index.html) dataset.

### Requirements

It requires the latest version of:

- pandas
- [pytorch](https://pytorch.org)
- [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [sentence-transformers](https://www.sbert.net/docs/installation.html)

It was tested for PyTorch 1.6.0 and CUDA 10.1.

### To Run

```none
python main.py -c <GPU Index>
```

### Results

```none
 [Epoch  98/100]  RMSE Loss (Train: 0.9086 | Test: 0.9261)
 [Epoch  99/100]  RMSE Loss (Train: 0.9092 | Test: 0.9264)
 [Epoch 100/100]  RMSE Loss (Train: 0.9087 | Test: 0.9261)
```

```none
 [Epoch  98/100]  RMSE Loss (Train: 0.9027 | Test: 0.9220)
 [Epoch  99/100]  RMSE Loss (Train: 0.9024 | Test: 0.9217)
 [Epoch 100/100]  RMSE Loss (Train: 0.9023 | Test: 0.9214)
```

```none
 [Epoch  98/100]  RMSE Loss (Train: 0.8965 | Test: 0.9250)
 [Epoch  99/100]  RMSE Loss (Train: 0.8967 | Test: 0.9246)
 [Epoch 100/100]  RMSE Loss (Train: 0.8967 | Test: 0.9246)
```
