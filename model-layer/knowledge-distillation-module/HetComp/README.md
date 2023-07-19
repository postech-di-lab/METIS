## Distillation from Heterogeneous Models for Top-K Recommendation
[![DOI](https://zenodo.org/badge/596085396.svg)](https://zenodo.org/badge/latestdoi/596085396)

This repository provides the source code of "Distillation from Heterogeneous Models for Top-K Recommendation" accepted in TheWebConf (WWW2023) as a research paper.


### 1. Overview
We present HetComp Framework that effectively compresses the valuable but difficult ensemble knowledge of heterogeneous models, generating a lightweight model with high recommendation performance. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/68782810/216048297-572b05a9-7010-4719-b51e-a32e27ba0eb5.png" width="100%"/>
<p>

### 2. Main Results
Training curves of w/o KD, DCD, and HetComp. Testing recall per 10 epochs. After convergence, we plot the last value.

#### 2-a. Benchmark setup
<p align="center">
    <img src="https://user-images.githubusercontent.com/68782810/216048646-2f81051e-ce52-4ae6-b3c6-7984552d867b.png" width="40%"/>
<img src="https://user-images.githubusercontent.com/68782810/216048663-3338be06-0d8a-4b58-80a6-316b279dfd89.png" width="40%"/>
<p>

#### 2-b. Generalization setup
<p align="center">
    <img src="https://user-images.githubusercontent.com/68782810/216048678-cb4c82bb-85ba-42a5-8c39-eff1d29e280f.png" width="40%"/>
<img src="https://user-images.githubusercontent.com/68782810/216048690-5e0ce142-84b3-4323-b231-8cdd73fe3364.png" width="40%"/>
<p>

We found that the sampling processes for top-ranked unobserved items are unnecessary, and removing the processes gave considerable performance improvements for the ranking matching KD methods (i.e., RRD, MTD, CL-DRD, and DCD). For this reason, we remove the sampling process for all ranking matching methods in our experiments. 


### 3. Requirements
#### 3-a. Dataset
- A-music dataset can be downloaded from: http://jmcauley.ucsd.edu/data/amazon/
- CiteULike dataset can be downloaded from: https://github.com/js05212/citeulike-t/blob/master/users.dat
- Foursquare dataset can be downloaded from: https://github.com/allenjack/SAE-NAD
#### 3-b. Software
- Python version: 3.6.10
- Pytorch version: 1.10.1
#### 3-c. Else.
- The target teacher models and their trajectories need to be located in Teachers directory.
- Due to its large size, we provide pretrained teacher trajectories through another file-sharing system: https://drive.google.com/file/d/1IYNJBbhzi2ETcKzruYOKFsJ__3RwyPJM/view?usp=share_link

