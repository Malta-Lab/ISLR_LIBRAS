#!/bin/bash
set -e
set -o pipefail

# videomae on SLOVO dataset with pre-train on kinetics and all best augmentations achieved on slovo dataset separately

# Ensure log directory exists
mkdir -p ./logs

# Set the seed value directly
seed=42

# Define the experiment name and log file
experiment_name="slovo_pretrain_all_best_augs_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=2,3,4 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -epochs 1000 \
    --gpus 3 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    --mixup \
    --mixup_alpha 0.25 \
    -t "normalize" "color_jitter" "random_rotation" "random_perspective" \
    -tp "color_jitter_0.4_0.4_0.4_0.4" "random_rotation_35" "random_perspective_0.5" \
    --seed "$seed" | tee -a "$log_file"

# Check if the process has started successfully
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Failed to complete the experiment ${experiment_name}." | tee -a "$log_file"
    exit 1
fi

echo "Experiment ${experiment_name} completed successfully." | tee -a "$log_file"
