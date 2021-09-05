ProxySR
=============

This is a PyTorch Implementation of ProxySR, proposed in "Unsupervised Proxy Selection for Session-based Recommender Systems", SIGIR'21.

## Configuation
Settings for training and evaluating ProxySR. Please refer to our paper for detailed description of each configuration. 
* \-\-dataset: Dataset. ex) diginetica
* \-\-batch_size: Mini-batch size for training. 
* \-\-val_batch_size: Mini-batch size for evaluation.
* \-\-embed_dim: Embedding size.
* \-\-lr: Learning rate.
* \-\-k: Number of proxies.
* \-\-dropout_rate: Dropout rate.
* \-\-margin: Margin for the marginal loss.
* \-\-lambda_dist: Regularization coefficient for distance regularizer.
* \-\-lambda_orthog: Regularization coefficient for orthogonality regularizer. 
* \-\-E: Number of annealing epoch.
* \-\-patience: Number of epoches to wait for learning to end after no improvement.
* \-\-max_position: Maximum length of input sequence.
* \-\-t0: Initial temperature. 
* \-\-te: Final temperature. 
* \-\-num_epoch: Maximum number of training epoches. 
* \-\-repetitive: (True) Next item recommendation with repetitive consumption or (False) Next unseen item recommendation.

## How to train and test a model on Diginetica dataset (next unseen item recommendation)
    python main.py --dataset=diginetica
    
## How to train and test a model on Diginetica dataset (item recommendation with repetitive consumption)
    python main.py --dataset=diginetica --repetitive=True

## Reference
* ### [Unsupervised Proxy Selection for Session-based Recommender Systems (SIGIR'21)](#)
  * [***Junsu Cho***](https://junsu-cho.github.io), SeongKu Kang, Dongmin Hyun, Hwanjo Yu
