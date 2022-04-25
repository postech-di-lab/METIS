# TaxoCom: A Framework for Topic Taxonomy Completion

- This is the author code of ["TaxoCom: Topic Taxonomy Completion with Hierarchical Discovery of Novel Topic Clusters (WWW 2022)"](https://to-be-appeared).
- This code is implemented based on the author code of ["TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering (KDD 2018)"](https://arxiv.org/abs/1812.09551) at [this repository](https://github.com/franticnerd/taxogen).

## Overview

<p align="center">
<img src="./figure/framework.png" width="1000">	
</p>

The overview of the TaxoCom framework which discovers the complete topic taxonomy by the recursive expansion of the given topic hierarchy. Starting from the root node, it performs **(1) locally discriminative embedding** and **(2) novelty adaptive clustering**, to selectively assign the terms (of each node) into one of the child nodes.

## Run the codes

#### STEP 1. Install the python libraries / packages

- `python`
- `numpy`, `scipy`
- [`spherecluster`](https://github.com/jasonlaska/spherecluster)
- `sklearn 0.21` (for the compatibility with `spherecluser`)

#### STEP 2. Download the dataset

- Download the datasets from the following links, then place them in `./data/nyt` and `./data/arxiv`, respectively.

  - [NYT dataset](https://drive.google.com/file/d/1UPoCLDyCDaP-_rWKfGSurNzY9DM0-OWJ/view?usp=sharing)
  - [arXiv dataset](https://drive.google.com/file/d/1wChAp6wyCR3ikXKpaYjuBJaLnDykzTJa/view?usp=sharing)

#### STEP 3. Execute the TaxoCom framework

- Run the codes by using the following commands
```
cd code
bash run_taxocom.sh <dataset-name> <seed-taxo-name>
```
- For example, the downloaded `nyt` directory can be simply used by
```
bash run_taxocom.sh nyt seed_taxo
```
