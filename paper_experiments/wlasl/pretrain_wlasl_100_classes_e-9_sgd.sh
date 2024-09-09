#!/bin/bash

experiment_name="WLASL_100_classes_e-9_sgd_seed_42"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=5 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -w 16 \
    -opt "sgd" \
    --gpus 1 \
    -sched "plateau" \
    -lr 0.000000001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "42" | tee -a "$log_file"

echo "Experiment ${experiment_name} completed." | tee -a "$log_file"
