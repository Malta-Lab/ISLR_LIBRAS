#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics using 10 seeds (no augmentations)

seed="42"

# Ensure log directory exists
mkdir -p ./logs

mu="XX"
cj="0.15"
rp="0.4"
rr="35"

experiment_name="agressive_splits_videomae_all_best_F1augs_${seed}"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=3,5 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 2 \
    -w 32 \
    -bs 16 \
    -epochs 900 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "minds" \
    -t "normalize" "color_jitter" "random_perspective" "random_rotation" \
    -tp "color_jitter_${cj}_${cj}_${cj}_${cj}" "random_perspective_${rp}" "random_rotation_${rr}" \
    --mixup \
    --mixup_alpha "$mu" \
    --seed "$seed" | tee -a "$log_file"

echo "All specified experiments completed." | tee -a "$log_file"
