#!/bin/bash

#SBATCH -J sl-fed-mm
#SBATCH -o demo-logs/sl-fed-mm.%j.out
#SBATCH -e demo-logs/sl-fed-mm.%j.err
#SBATCH -t 72:00:00              

#### Select  GPU
#SBATCH -p RTX4090
#SBATCH --gres=gpu:1

##  node 지정하기
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --exclude=n52,n53,n54,n[77-80]

cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh

# echo "conda activate llm4ts"
# conda activate llm4ts

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

    
# echo "deactivate llm4ts"
# conda deactivate

date
squeue --job $SLURM_JOBID

echo  "##### END #####"

