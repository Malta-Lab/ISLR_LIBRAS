#!/bin/bash
# VideoMAE on MINDS dataset with fine-tuning from Slovo checkpoints, running with a single seed (42).

SEED=42

# Ensure log directory exists
mkdir -p ./logs

# Define the experiment name and log file
experiment_name="new_slovo_minds_finetune_$SEED"
log_file="./logs/${experiment_name}.log"

# Define the path to the Slovo checkpoint corresponding to the current seed
checkpoint_path="/mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/slovo/base/slovo_pretrain_seed_44/version_0/checkpoints/best_model.ckpt"

echo "Running ${experiment_name} with seed $SEED" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=1,3,4,5 python train_v2.py \
    --finetune "$checkpoint_path" \
    --gpus 4 \
    -w 8 \
    -epochs 2000 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "minds" \
    -t "normalize" \
    --seed "$SEED" 2>> "$log_file" | tee -a "$log_file"

# Check if the process has started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
    exit 1
fi

# Introduce a short sleep to avoid overloading I/O
sleep 2

echo "Experiment with seed $SEED completed." | tee -a "$log_file"

