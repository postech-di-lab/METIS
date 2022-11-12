Contrastive Learning: Relaxed Contextualized word Mover Distance (CLRCMD)
==================

This repository reproduces the experimental result of CLRCMD (pronounced as "clear command") reported in [the paper](https://arxiv.org/abs/2202.13196) to be appeared in ACL 2022 main track.

## 0. Download checkpoints
We want to upload our checkpoint to model registry such as huggingface hub to make them easily accessible, but due to the complicated process, we decided to just manually upload the checkpoint to our gdrive.
Please visit this [link](https://drive.google.com/drive/folders/1q-a7z2Xy09dThp3FtCVdH2GprcEykgaa?usp=sharing) to download the checkpoints we used in our experiment.
We assume the `pytorch_model.bin` is located in `/home/username/checkpoints/bert-rcmd/pytorch_model.bin`

## 1. Prepare Environment
We assume that the user uses anaconda environment.
```
conda create -n clrcmd python=3.8
conda activate clrcmd
pip install -r requirements.txt
python setup.py develop
```

## 2. Prepare dataset

### 2-1. Semantic Textual Similarity benchmark (STS12, STS13, STS14, STS15, STS16, STSB, SICKR)
We download the benchmark dataset using the script provided by SimCSE repository.  
```
bash examples/download_sts.sh
```
* `tokenizer.sed`: Tokenizer script used in `download_sts.bash`

### 2-2. Interpretable Semantic Textual Similarity benchmark (iSTS)
We create a script for downloading iSTS benchmarks.
```
bash examples/download_ists.sh
```

#### 2-2-1. Correct wrong input format
* `STSint.testinput.answers-students.sent1.chunk.txt`
 * 252th example: from `a closed path` to `a closed path.`
 * 287th example: from `has no gaps` to `[ has no gaps ]`
 * 315th example: from `is in a closed path,` to `[ is in a closed path, ]`
 * 315th example: from `is in a closed path.` to `[ is in a closed path. ]`
* `STSint.testinput.answers-students.sent1.txt`
 * 287th example: `battery  terminal` to `battery terminal`
 * 308th example: `switch z,  that` to `switch z, that`
* `STSint.testinput.answers-students.sent2.chunk.txt`
 * 287th example: `are not separated by the gap` to `[ are not separated by the gap ]`
 * 315th example: `are` to `[ are ]`
 * 315th example: `in closed paths` to `[ in closed path ]`

### 2-3. NLI dataset tailored for self-supervised learning (SimCSE-NLI)
We download the training dataset using the script provided by SimCSE repository.
```
bash examples/download_nli.bash
```
