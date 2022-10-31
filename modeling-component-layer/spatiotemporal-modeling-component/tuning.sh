#!/bin/bash
data=$1
gpunum=$2
resultpath="results/""$model""/$data""/"

mkdir -p "$resultpath"

declare -a lrs=("5e-3" "1e-3")
declare -a bss=("128" "256")

declare -a lbs=("0.3" "0.5" "0.7")
declare -a mus=("0.3" "0.5" "0.7")

declare -a tus=("0.5") # 0 can also be tried


for lr in "${lrs[@]}"
do

for bs in "${bss[@]}"
do

for lb in "${lbs[@]}"
do

for mu in "${mus[@]}"
do

for tu in "${tus[@]}"
do

python train.py --dataset "$data" --gpu "$gpunum" --learning_rate "$lr" --batch_size "$bs" --lamb "$lb" --mu "$mu" --tau "$tu" &> "$resultpath"result_lr"$lr"_bs"$bs"_lb"$lb"_mu"$mu"_tau"$tu".txt 

done

done

done

done

done