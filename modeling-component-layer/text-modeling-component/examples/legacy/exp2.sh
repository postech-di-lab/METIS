checkpoint_dir_list=(
    "/home/sh0416/checkpoints/20211029_101219/"
    "/home/sh0416/checkpoints/20211030_161612/"
)
checkpoint_path_list=(
    "/home/sh0416/checkpoints/20211029_101219/checkpoint-2000/pytorch_model.bin"
    "/home/sh0416/checkpoints/20211030_161612/checkpoint-2000/pytorch_model.bin"
)
source_list=("answers-students")

for source in ${source_list[@]}; do
    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211029_101219/ \
        --ckpt-path /home/sh0416/checkpoints/20211029_101219/checkpoint-2000/pytorch_model.bin

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211029_101219/

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_161612/ \
        --ckpt-path /home/sh0416/checkpoints/20211030_161612/checkpoint-2000/pytorch_model.bin

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_161612/

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_190839/ \
        --ckpt-path /home/sh0416/checkpoints/20211030_190839/checkpoint-1000/pytorch_model.bin

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_190839/

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_191325/ \
        --ckpt-path /home/sh0416/checkpoints/20211030_191325/checkpoint-4000/pytorch_model.bin

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir /home/sh0416/checkpoints/20211030_191325/
done