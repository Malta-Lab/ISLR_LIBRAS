#!/bin/bash

# Train only classifier of videomae on MINDS dataset
mkdir -p ./logs

# Define the list of seeds
seeds=(42)

i=1
for seed in "${seeds[@]}"
do
    experiment_name="videomae_kinetics_freeze_10k_epochs_no_augs_${seed}"
    log_file="./logs/${experiment_name}.log"

    echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --freeze \
        -epochs 10000 \
        --gpus 1 \
        -w 16 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --dataset "minds" \
        -t "normalize" \
        --seed "$seed" | tee -a "$log_file"

    if [ $? -ne 0 ]; then
        echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    sleep 2

    ((i++))
    if [ $i -gt 10 ]; then
        break
    fi
done

echo "All specified experiments completed." | tee -a "$log_file"
