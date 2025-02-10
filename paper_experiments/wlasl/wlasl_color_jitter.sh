#!/bin/bash
# videomae on WLASL dataset with pre-train on kinetics and color jitter 0.5

seed=42

# Define the experiment name and log file
experiment_name="WLASL_color_jitter_0.5_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed and color jitter 0.5" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=1 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 1 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "color_jitter" "normalize" \
    -tp "color_jitter_0.5_0.5_0.5_0.5" \
    --seed "$seed" | tee -a "$log_file"

echo "Experiment with color jitter 0.5 complete." | tee -a "$log_file"