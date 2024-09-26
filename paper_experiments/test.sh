#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics, testing the experiment that got .665 accuracy

SEED=42

experiment_name="test_acc_.622_$SEED"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=1,3 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 2 \
    -sched "plateau" \
    -epochs 1500 \
    -lr 0.0001 \
    --data_path "../MINDS_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    -bs 16 \
    --dataset "minds" \
    -t "normalize" "color_jitter" "random_perspective" "random_horizontal_flip" "gaussian_blur" \
    -tp "color_jitter_0.5_0.5_0.5_0.5" "random_perspective_0.5"  \
    --seed "$SEED" | tee -a "$log_file"

echo "Experiment completed." | tee -a "$log_file"

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "no_pretrain_videomae" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16
