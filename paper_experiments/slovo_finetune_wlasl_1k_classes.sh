#!/bin/bash
# using VIDEOMAE trained on WLASL 1k classes to finetune on SLOVO dataset
experiment_name="slovo_finetune_WLASL_1k_classes_seed_42_new"
log_file="./logs/${experiment_name}.log"

checkpoint_path="/mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/wlasl/WLASL_1k_classes_pretrain_seed_42/version_0/checkpoints/best_model.ckpt"

# Run the training script
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train.py \
    --finetune $checkpoint_path \
    -w 16 \
    --gpus 5 \
    -epochs 2500 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "slovo" \
    -t "normalize" \
    --seed "42" | tee -a "$log_file"

echo "Experiment ${experiment_name} completed." | tee -a "$log_file"
