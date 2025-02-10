#!/bin/bash
# videomae on SLOVO dataset with pre-train on kinetics and mixup 0.25

seed=42

# Define the experiment name and log file
experiment_name="slovo_mixup_0.25_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name}" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=2,3 python train_v3.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 2 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    -t "normalize" \
    --mixup \
    --mixup_alpha 0.25 \
    --seed "$seed" | tee -a "$log_file"

echo "Experiment with mixup 0.25 complete." | tee -a "$log_file"
