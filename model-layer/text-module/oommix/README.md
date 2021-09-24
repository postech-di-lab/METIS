Out-of-manifold Regularization in Contextual Embedding Space for Text Classification
====================

Appeared in ACL2021 main conference  
Arxiv: [https://arxiv.org/abs/2105.06750](https://arxiv.org/abs/2105.06750)

## Download data source
1. Download tarfile from the [link](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?usp=sharing)
1. Extract tarfile

The data directory tree looks like this.
```
.
├── ag_news_csv
│   ├── classes.txt
│   ├── eda_train.csv
│   ├── readme.txt
│   ├── test.csv
│   └── train.csv
├── ag_news_csv.tar.gz
├── amazon_review_polarity_csv
│   ├── readme.txt
│   ├── test.csv
│   └── train.csv
├── amazon_review_polarity_csv.tar.gz
├── dbpedia_csv
│   ├── classes.txt
│   ├── readme.txt
│   ├── test.csv
│   └── train.csv
├── dbpedia_csv.tar.gz
├── yahoo_answers_csv
│   ├── classes.txt
│   ├── readme.txt
│   ├── test.csv
│   └── train.csv
└── yahoo_answers_csv.tar.gz
```
I will call the path of root data directory ${DATA_DIR}.

## How to execute

```
virtualenv ~/venv/oommix --python=/usr/bin/python3
source ~/venv/oommix/bin/activate
pip install -r requirements.txt
python main.py --data_dir ${DATA_DIR}/ag_news_csv
```

## Help message

```
usage: main.py [-h] [--seed SEED] [--gpu GPU] --data_dir DATA_DIR
               [--dataset {ag_news,yahoo_answer,amazon_review_polarity,dbpedia}]
               [--num_train_data NUM_TRAIN_DATA]
               [--data_augment {none,eda,backtranslate,ssmba}]
               [--max_length MAX_LENGTH] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--lr LR] [--drop_prob DROP_PROB]
               [--mix_strategy {none,tmix,nonlinearmix,mixuptransformer,oommix}]
               [--m_layer M_LAYER] [--d_layer D_LAYER] [--alpha ALPHA]
               [--coeff_intr COEFF_INTR] [--eval_every EVAL_EVERY]
               [--patience PATIENCE]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed (default: 1)
  --gpu GPU             Index of gpu (default: 0)

data:
  --data_dir DATA_DIR   Data directory (default: None)
  --dataset {ag_news,yahoo_answer,amazon_review_polarity,dbpedia}
                        Dataset (default: ag_news)
  --num_train_data NUM_TRAIN_DATA
                        Number of train dataset. Use the first
                        `num_train_data` row. -1 means whole dataset (default:
                        -1)
  --data_augment {none,eda,backtranslate,ssmba}
                        Data augmentation technique (default: none)
  --max_length MAX_LENGTH
                        Maximum length for transformer input (default: 256)

train:
  --epoch EPOCH         Number of epochs (default: 5000)
  --batch_size BATCH_SIZE
                        Batch size (default: 12)
  --lr LR               Learning rate (default: 2e-05)
  --drop_prob DROP_PROB
                        Dropout probability (default: 0.1)
  --mix_strategy {none,tmix,nonlinearmix,mixuptransformer,oommix}
                        Mixup strategy during training (default: none)
  --m_layer M_LAYER     Embedding generator layer (default: 3)
  --d_layer D_LAYER     Manifold discriminator layer (default: 12)
  --alpha ALPHA         Parameter for beta distribution (default: 0.2)
  --coeff_intr COEFF_INTR
                        Coefficient for intrusion objective (default: 0.5)
  --eval_every EVAL_EVERY
                        Period step for reporting evaluation metric (default:
                        200)
  --patience PATIENCE   Training stops until `patience` evaluations doesn't
                        increase (default: 10)
```
