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

## 3. Conduct experiments

### 3-1. Evaluate semantic textual similarity benchmark without any training
```
# Help message
python -m examples.run_evaluate_sts -h

# One example
python -m examples.run_evaluate_sts --data-dir data --model bert-rcmd
```

### 3-2. Train model using self-supervised learning (e.g. SimCSE, CLRCMD)
```
python -m examples.run_train --data-dir data --model bert-rcmd
```

### 3-2. Evaluate benchmark performance on the trained checkpoint
```
python -m examples.run_evaluate_sts --data-dir data --model bert-rcmd --checkpoint /home/username/checkpoints/bert-rcmd
```

### 3-3. Evaluate interpretable semantic textual similarity benchmark
```
# Filter out the alignments which has low score
python -m examples.run_preprocess_ists --alignment-path data/ISTS/test_goldStandard/STSint.testinput.images.wa

# Bert-avg
python -m examples.run_evaluate_ists --data-dir data/ISTS/test_goldStandard/ --source images --checkpoint-dir checkpoints/bert-avg/
./data/ISTS/test_goldStandard/evalF1.pl ./data/ISTS/test_goldStandard/STSint.testinput.images.wa.equi ./checkpoints/bert-avg/images.wa

# Bert-Clrcmd
python -m examples.run_evaluate_ists --data-dir data/ISTS/test_goldStandard/ --source images --checkpoint-dir checkpoints/bert-rcmd/
./data/ISTS/test_goldStandard/evalF1.pl ./data/ISTS/test_goldStandard/STSint.testinput.images.wa.equi ./checkpoints/bert-rcmd/images.wa
```

## 4. Report results

### 4-1. Semantic textual similarity benchmark
|checkpoint|sts12|sts13|sts14|sts15|sts16|stsb|sickr|avg|
|----------|-----|-----|-----|-----|-----|----|-----|---|
|`bert-rcmd`|0.7523|0.8506|0.8099|0.8626|0.8150|0.8521|0.8049|0.8211|
