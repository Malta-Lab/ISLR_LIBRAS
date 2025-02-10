#!/bin/bash

experiment_name="WLASL_100_classes_pretrain_seed_42_new"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=2,3,4,5 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -epochs 2500 \
    -w 16 \
    --gpus 4 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32_complete" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "42" | tee -a "$log_file"

echo "Experiment ${experiment_name} completed." | tee -a "$log_file"
