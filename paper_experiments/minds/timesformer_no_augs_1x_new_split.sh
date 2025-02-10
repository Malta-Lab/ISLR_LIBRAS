#!/bin/bash
# timesformer on MINDS dataset, no pre-train on kinetics, no augs, with 10 seeds from seeds.txt

seed='42'

# Ensure log directory exists
mkdir -p ./logs

experiment_name="new_splits_timesformer_no_augmentations_${seed}"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=0 python train.py \
    -ptm "facebook/timesformer-base-finetuned-k400" \
    --gpus 1 \
    --no_pretrain \
    -epochs 900 \
    -w 8 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "minds" \
    -t "normalize" \
    --seed "$seed" | tee -a "$log_file"

echo "All specified experiments completed." | tee -a "$log_file"