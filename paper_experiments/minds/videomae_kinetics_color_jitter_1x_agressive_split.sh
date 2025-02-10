#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics with 10 color jitter values

# Define array of color jitter values (degrees)
color_jitter=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# Ensure log directory exists
mkdir -p ./logs

# Fixed seed value
seed=42

for cj in "${color_jitter[@]}"
do
    experiment_name="agressive_splits_videomae_kinetics_color_jitter_${cj}_${seed}"
    log_file="./logs/${experiment_name}.log"

    echo "Starting experiment ${experiment_name}..." | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=2,3 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 2 \
        -w 16 \
        -bs 16 \
        -epochs 900 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --random_sample \
        --dataset "minds" \
        -t "color_jitter" "normalize" \
        -tp "color_jitter_${cj}_${cj}_${cj}_${cj}" \
        --seed "$seed" | tee -a "$log_file"

    # Check if the process failed
    if [ $? -ne 0 ]; then
        echo "Failed to complete experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    # Short sleep to avoid I/O overload
    sleep 1
done

echo "All experiments completed." | tee -a "$log_file"