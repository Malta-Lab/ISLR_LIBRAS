#!/bin/bash
# videomae on SLOVO dataset with pre-train on kinetics and no augmentations 10x (1 per seed)

seed=2

# Define the experiment name and log file
experiment_name="slovo_pretrain_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed and starting lr 1e-2" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=2 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 1 \
    -sched "plateau" \
    -lr 0.01 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    -t "normalize" \
    --seed "$seed" 2>> "$log_file" | tee -a "$log_file"

echo "Experiment with starting lr 1e-2 complete." | tee -a "$log_file"
