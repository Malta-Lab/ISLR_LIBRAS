#!/bin/bash
# videomae on WLASL dataset with pre-train on kinetics, testing new models_v2.py file which save metrics (using torchmetrics) on the validation end and correct split of the WLASL dataset

SEED=16

experiment_name="test_experiment_new_models_v2_seed_${SEED}_corect_split"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=4,5 python train_v3.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    --gpus 2 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "../WLASL_tensors_32" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "$SEED" | tee -a "$log_file"

echo "Experiment completed." | tee -a "$log_file"
