#!/bin/bash
# videomae on SLOVO dataset with pre-train on kinetics and color jitter 0.5

seed=42

# Define the experiment name and log file
experiment_name="slovo_color_jitter_0.5_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed and color jitter 0.5" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=2,3,4,5 python train_v3.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 4 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    -t "color_jitter" "normalize" \
    -tp "color_jitter_0.5_0.5_0.5_0.5" \
    --seed "$seed" | tee -a "$log_file"

echo "Experiment with color jitter 0.5 complete." | tee -a "$log_file"
