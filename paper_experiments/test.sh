#!/bin/bash
# videomae on WLASL dataset with pre-train on kinetics, testing new models_v2.py file which save metrics (using torchmetrics) on the validation step

SEED=2

experiment_name="new_dataset_v2_all_best_augs_1500epochs_patience50_experiment_seed_$SEED"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=1,3,4,5 python train.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 4 \
    -sched "plateau" \
    -epochs 1500 \
    -lr 0.0001 \
    --data_path "../WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    --mixup \
    --mixup_alpha 0.25 \
    -t "normalize" "color_jitter" "random_rotation" "random_perspective" \
    -tp "color_jitter_0.5_0.5_0.5_0.5" "random_rotation_0.35" "random_perspective_0.5" \
    --seed "$SEED" | tee -a "$log_file"

echo "Experiment completed." | tee -a "$log_file"
