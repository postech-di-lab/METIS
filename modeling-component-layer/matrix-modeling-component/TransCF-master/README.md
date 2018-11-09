# Translational Collaborative Metric Learning (TransCF)

### Overview
> Recently, matrix factorization-based recommendation methods have been criticized for the problem raised by the triangle inequality violation. Although several metric learning-based approaches have been proposed to overcome this issue, existing approaches typically project each user to a single point in the metric space, and thus do not suffice to properly model the *intensity* and the *heterogeneity* of user-item relationships of implicit feedback. In this paper, we propose TransCF to discover such latent user-item relationships embodied in implicit user-item interactions. Inspired by the translation mechanism popularized by knowledge graph embedding, we construct user-item specific translation vectors by employing the neighborhood information of users and items, and translate each user towards items regarding the user's relationships with the items.

### Paper
- Translational Collaborative Metric Learning (*ICDM 2018*)
  - [_**Chanyoung Park**_](http://di.postech.ac.kr/~pcy1302), Donghyun Kim, Xing Xie, Hwanjo Yu

### Requirements

- Python version: 2.7
- Pytorch version: 0.3.0a0+669a99b
- fastrand (Fast random number generation in Python)
  - [See installation instructions](https://github.com/pcy1302/fastrand)
- Multicore-TSNE (only required for t-SNE visualization)
  - [See installation instructions](https://github.com/pcy1302/Multicore-TSNE)
  

### How to Run

```
git clone https://github.com/pcy1302/TransCF.git
cd TransCF
python main.py --recommender TransCF --dataset delicious --lRate 0.005 --mode Val
```

### Configuration
You can evaluate TransCF with different settings. Below is a description of all the configurable parameters:

- --recommender : 'Choose a recommender.'
- --dataset : 'Choose a dataset.' (delicious, bookcrossing, ciao, cellphone)
- --embedding_dim : 'Number of embedding dimensions.'
- --lRate :  'Learning rate.'
- --margin' : 'Margin.'
- --reg1' : 'Distance regularizer.'
- --reg2' : 'Neighborhood regularizer.'
- --mode : 'Validation or Test'	(Val, Test)
- --numEpoch : 'Number of epochs.'
- --num_negatives' : 'Number of negative samples.
- --batch_size' : 'Batch size.'
- --rand_seed' : 'Random seed.'
- --cuda : 'Speficy GPU number'
- --early_stop : 'Early stop iteration.'


### Reproducing the qualitative experiments 
- [Tables 4 and 6](https://github.com/pcy1302/TransCF/blob/master/Qualitative_Intensity_Table_4_6.ipynb) 
- Table 5 and 7, Figures 4 and 5
  - [Ciao dataset](https://github.com/pcy1302/TransCF/blob/master/Qualitative_Ciao.ipynb) 
  - [Amazon C&A dataset](https://github.com/pcy1302/TransCF/blob/master/Qualitative_Amazon.ipynb)

