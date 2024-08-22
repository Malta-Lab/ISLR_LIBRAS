#!/bin/bash
# videomae on SLOVO dataset with pre-train on kinetics and no augmentations 10x (1 per seed)

seed=44

# Define the experiment name and log file
experiment_name="scheduler_ES50_MAXEPOCHS600_slovo_pretrain_seed_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed and LinearLR scheduler with AdamW optimizer" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=3 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 1 \
    -sched "linearlr" \
    -epochs 600 \
    -lr 0.001 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    -t "normalize" \
    --seed "$seed" 2>> "$log_file" | tee -a "$log_file"

echo "Experiment with LinearLR and AdamW complete." | tee -a "$log_file"
