#!/bin/bash
#Videomae pretrained on kinetics training on WLASL 1k datasetBRACIS-2024/paper_experiments/wlasl/pretrain_wlasl_1k_classes_test.sh
experiment_name="WLASL_1k_classes_pretrain_seed_42"
log_file="./logs/${experiment_name}.log"

# Run the training script
CUDA_VISIBLE_DEVICES=1,3,4,5 python train_v2.py \
    -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
    -w 16 \
    --gpus 4 \
    -epochs 2000 \
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
