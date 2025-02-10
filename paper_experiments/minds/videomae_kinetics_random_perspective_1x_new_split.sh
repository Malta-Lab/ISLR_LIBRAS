#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with 10 random perspective

seed="42"

# Define array of random perspective values (degrees)
random_perspective=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# Ensure log directory exists
mkdir -p ./logs

# Loop through each random perspective value
for rp in "${random_perspective[@]}"
do
    experiment_name="new_splits_videomae_kinetics_random_perspective_${rp}_${seed}"
    log_file="./logs/${experiment_name}.log"

    echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=3 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        -w 8 \
        --gpus 1 \
        -sched "plateau" \
        -epochs 900 \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --random_sample \
        --dataset "minds" \
        -t "random_perspective" "normalize" \
        -tp "random_perspective_${rp}" \
        --seed "$seed" | tee -a "$log_file"

    # Check if the process has started successfully
    if [ $? -ne 0 ]; then
        echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    # Introduce a short sleep to avoid overloading I/O
    sleep 1
done

echo "All specified experiments completed." | tee -a "$log_file"