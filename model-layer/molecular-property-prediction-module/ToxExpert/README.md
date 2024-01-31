# Learning Topology-Specific Experts for Molecular Property Prediction

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://aaai.org/Conferences/AAAI-23/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'23-brightgreen" /></a>
</p>


This is Official Pytorch Implementation for the paper "Learning Topology-Specific Experts for Molecular Property Prediction". Suyeon Kim, Dongha Lee, SeongKu Kang, Seonghyeon Lee, Hwanjo Yu **(AAAI-23)**

The paper is available at [Link](https://arxiv.org/abs/2302.13693).

<p align="center">
  <img src="https://github.com/kimsu55/ToxExpert/blob/main/img/fig3_main_arch.jpg" width="500" title="The overall framework of TopExpert">
</p>

## Run  
```
python main.py --dataset bbbp
```

We refer the baseline code to build our implementation.
[https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)

## Package Install

``` python  

conda create -n topexpert python=3.8

conda activate topexpert

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

conda install -c conda-forge rdkit

conda install pytorch-geometric -c rusty1s -c conda-forge

```  

## Cite (Bibtex)
- If you find ``TopExpert`` useful in your research, please consider citing:

```
@article{kim2023learning,
  title={Learning Topology-Specific Experts for Molecular Property Prediction},
  author={Suyeon Kim, Dongha Lee, SeongKu Kang, Seonghyeon Lee, Hwanjo Yu},
  booktitle={AAAI},
  year={2023}
}
```



