## Consensus Learning from Heterogeneous Objectives for One-Class Collaborative Filtering

This repository provides the source code of "Consensus Learning from Heterogeneous Objectives for One-Class Collaborative Filtering" accepted in TheWebConf (WWW2022) as a research paper.

### 1. Overview
We present ConCF Framework that exploits the complementarity from heterogeneous learning objectives throughout the training process, generating a more generalizable model.

<p align="center">
    <img src="https://user-images.githubusercontent.com/68782810/150118110-95996faa-7828-4bf4-aa1d-76d4c94b469d.png" width="550"/>
<p>

In this work, we use five learning objectives for OCCF that have been widely adopted in recent work.
- CF-A: Bayesian Personalized Ranking (BPR) Loss
- CF-B: Collaborative Metric Learning (CML) Loss
- CF-C: Binary Cross-Entropy (BCE) Loss
- CF-D: Mean Squared Error (MSE) Loss
- CF-E: Multinomial Likelihood Loss


### 2. Main Results
#### 2-a. SingleCF vs. ConCF
- The performance of the head trained by CF-A is significantly improved in ConCF.
- The consensus collaboratively evolves with the heads based on their complementarity, providing accurate supervision..

  ![image](https://user-images.githubusercontent.com/68782810/150125272-07116093-e483-4ebe-aeeb-c17ed585324a.png)

#### 2-b. ConCF (w/o CL) vs. ConCF
- ConCF (w/o CL) cannot effectively improve the performance of each head.

  ![image](https://user-images.githubusercontent.com/68782810/150125321-b4e6522a-f834-4b39-b552-e8348278707d.png)


### 3. Requirements
- Python version: 3.6.10
- Pytorch version: 1.5.0

### 4. How to Run
```
Please refer to 'Guide to using ConCF.ipynb' file.
```
