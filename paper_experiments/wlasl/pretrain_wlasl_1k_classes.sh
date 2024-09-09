#!/bin/bash

experiment_name="WLASL_1k_classes_pretrain_seed_42"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=1,3 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -w 16 \
    --gpus 2 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "42" | tee -a "$log_file"

echo "All specified experiments completed." | tee -a "$log_file"
