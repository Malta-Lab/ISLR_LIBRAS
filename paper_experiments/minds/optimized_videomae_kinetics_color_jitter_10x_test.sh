#!/bin/bash
# Script for multi-GPU training using optimized VideoMAE with shared memory.
# Runs experiments with varying color jitter values on the selected dataset (e.g., MINDS).
# Increase file descriptor limit to avoid issues with many .pt files.

seed='42'

# Define an array of color jitter values.
color_jitter=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# Create log directory.
mkdir -p ./logs

for cj in "${color_jitter[@]}"
do
    experiment_name="optimized_videomae_kinetics_color_jitter_${cj}_${seed}"
    log_file="./logs/${experiment_name}.log"

    echo "Running experiment ${experiment_name}..." | tee -a "$log_file"

    # Use GPUs 2,3,4,5 (adjust as needed).
    CUDA_VISIBLE_DEVICES=2 python train_optimized.py \
        --pretrained_model "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 1 \
        --workers 4 \
        -bs 16 \
        --max_epochs 900 \
        --learning_rate 0.0001 \
        --scheduler "plateau" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --dataset "minds" \
        --transforms "color_jitter" \
        -tp "color_jitter_${cj}_${cj}_${cj}_${cj}" \
        --seed "$seed" \
        --patience 30 | tee -a "$log_file"

    if [ $? -ne 0 ]; then
        echo "Experiment ${experiment_name} failed!" | tee -a "$log_file"
        exit 1
    fi

    sleep 2
done

echo "All experiments completed successfully." | tee -a "$log_file"
