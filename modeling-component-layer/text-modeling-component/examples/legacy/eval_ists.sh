source_list=("images" "headlines")

ckpt_dir=""
ckpt_path=""

for source in ${source_list[@]}; do
    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir ${ckpt_dir} \
        --ckpt-path ${ckpt_path}

    PYTHONPATH=src python src/scripts/run_ists.py \
        --source ${source} \
        --ckpt-dir ${ckpt_dir}
done