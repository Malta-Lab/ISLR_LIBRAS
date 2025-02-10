#!/bin/bash
# videomae on MINDS dataset with single seed (42) and mixup variations

# Define array of mixup values (degrees)
mixup=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Fixed seed value
seed=42

# Ensure log directory exists
mkdir -p ./logs

for mu in "${mixup[@]}"
do
    experiment_name="videomae_kinetics_mixup_${mu}_${seed}"
    log_file="./logs/${experiment_name}.log"
    
    echo "Running experiment ${experiment_name}..." | tee -a "$log_file"
    
    CUDA_VISIBLE_DEVICES=3 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 1 \
        -w 4 \
        -epochs 900 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --random_sample \
        --dataset "minds" \
        -t "normalize" \
        --mixup \
        --mixup_alpha "$mu" \
        --seed "$seed" | tee -a "$log_file"

    # Check if the process completed successfully
    if [ $? -ne 0 ]; then
        echo "Failed to complete experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    # Introduce a short sleep to avoid overloading I/O
    sleep 1
done

echo "All experiments for seed ${seed} completed." | tee -a "$log_file"