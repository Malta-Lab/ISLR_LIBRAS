#!/bin/bash
# starting LR of 1e-8, sched patience 10, ES 50

seed=42

# Define the experiment name and log file
experiment_name="WLASL_LR_1e-8_$seed"
log_file="./logs/${experiment_name}.log"

echo "Running ${experiment_name} with seed $seed" | tee -a "$log_file"

# Run the training script
CUDA_VISIBLE_DEVICES=1,3,4,5 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 4 \
    -sched "plateau" \
    -lr 0.00000001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "$seed" | tee -a "$log_file"

echo "Experiment with LR 1e-8 complete." | tee -a "$log_file"