#!/bin/bash
#Videomae pretrained on kinetics training on WLASL 1k classes (complete dataset)
experiment_name="WLASL_1k_classes_pretrain_seed_42"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train_v2.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -w 16 \
    --gpus 5 \
    -epochs 2500 \
    -sched "plateau" \
    -lr 0.0001 \
    --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32_complete" \
    --exp_name "$experiment_name" \
    --frames 16 \
    --random_sample \
    --dataset "wlasl" \
    -t "normalize" \
    --seed "42" | tee -a "$log_file"

echo "All specified experiments completed." | tee -a "$log_file"
