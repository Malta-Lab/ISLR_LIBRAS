#!/bin/bash

# early stop of 100 epochs

experiment_name="WLASL_300_classes_all_best_augs_pretrain_seed_42"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train_v2.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -w 16 \
    --gpus 5 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32_complete" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    --mixup \
    --mixup_alpha 10 \
    -t "normalize" "color_jitter" "random_perspective" "random_rotation"\
    -tp "random_perspective_0.5" "random_rotation_40" "color_jitter_0.5_0.5_0.5_0.5" \
    --seed "42" | tee -a "$log_file"

echo "All specified experiments completed." | tee -a "$log_file"
