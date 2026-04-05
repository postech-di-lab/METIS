# Multimodal F³CRec: Federated Continual Recommendation with Early Fusion

This repository provides the PyTorch implementation of the Multimodal Federated Continual Recommendation system. It extends the foundational F³CRec framework (Lim et al., **CIKM 2025**) by integrating CLIP-based visual and textual embeddings to mitigate the cold-start problem and catastrophic forgetting during incremental learning phases.

## 📌 Requirements

```bash
# Create the isolated environment from the configuration file
conda env create -f environment.yml

# Activate the environment
conda activate fcrec

# Go to the project directory
cd fcrec
```

## 🚀 Running the Experiments

The training pipeline is divided into two strict phases to maintain the stability-plasticity balance in continual learning.

### Step 1: Base Task Training

```bash
python main.py \
    --save_model 1 \
    --load_model 0 \
    --backbone fedmf \
    --model fcrec \
    --lr 1.0 \
    --dim 32 \
    --patience 30 \
    --client_cl \
    --server_cl \
    --reg_client_cl 0.1 \
    --eps 0.006 \
    --topN 30 \
    --beta 0.9 \
    --num_round 100 \
    --dataset ml-100k \
    --mode concat \
    --clip_dim 1024 \
    --alpha 0.3 \
    --proj_dim 128 \
    --proj_lr_ratio 0.1 \
    --vision_embedding_path ./embeddings/movie_image_embeddings_512.pt \
    --text_embedding_path ./emgeddings/movie_text_embeddings_512.pt \
```

### Step 2: Incremental Task Training

```bash
python main.py \
    --load_model 1 \
    --backbone fedmf \
    --model fcrec \
    --lr 1.0 \
    --dim 32 \
    --patience 30 \
    --client_cl \
    --server_cl \
    --reg_client_cl 0.1 \
    --eps 0.006 \
    --topN 30 \
    --beta 0.9 \
    --num_round 100 \
    --dataset ml-100k \
    --mode concat \
    --clip_dim 1024 \
    --alpha 0.3 \
    --proj_dim 128 \
    --proj_lr_ratio 0 \
    --vision_embedding_path ./embeddings/movie_image_embeddings_512.pt \
    --text_embedding_path ./embeddings/movie_text_embeddings_512.pt
```

## 📊 Evaluation
The model evaluates its ranking performance using **NDCG@20 (N@20)** and **Recall@20 (R@20)** metrics on the test sets of current and past tasks to measure knowledge retention. N@20 is the primary metric, as it strictly evaluates the positional alignment of the multimodal fusion.