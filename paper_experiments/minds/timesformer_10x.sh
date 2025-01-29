#!/bin/bash

# Train TimeSformer on MINDS dataset
SEEDS_FILE="./seeds.txt"
mkdir -p ./logs

i=1
while IFS= read -r seed
do

experiment_name="timesformer_no_augs_${seed}"
log_file="./logs/${experiment_name}.log"

echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train.py \
            -ptm "facebook/timesformer-base-finetuned-k400" \
            --no_pretrain \
            --gpus 5 \
            -w 16 \
            -sched "plateau" \
            -lr 0.0001 \
            --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
            --exp_name "$experiment_name" \
            --frames 16 \
            --random_sample \
            --dataset "minds" \
            -t "normalize" \
            -bs 8 \
            --seed "$seed" | tee -a "$log_file"

if [ $? -ne 0 ]; then
    echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
    exit 1
fi

sleep 2

((i++))
if [ $i -gt 10 ]; then
    break
fi
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"