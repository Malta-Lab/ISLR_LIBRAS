#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics (single seed 42) with 10 random rotations

# Define seed
seed=42

# Define array of random rotation values (degrees)
random_rotation=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Ensure log directory exists
mkdir -p ./logs

for rr in "${random_rotation[@]}"
do
    # Check if both best.ckpt and last.ckpt files exist in the version_X/checkpoints directories
    both_ckpt_files_exist=false
    for dir in /mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/videomae_kinetics_random_rotation_"${rr}"_"${seed}"/version_*/checkpoints; do
        if [ -f "$dir/best_model.ckpt" ] && [ -f "$dir/top5_best_model.ckpt" ]; then
            both_ckpt_files_exist=true
            break
        fi
    done

    experiment_name="videomae_kinetics_random_rotation_${rr}_${seed}"
    log_file="./logs/${experiment_name}.log"
    
    if [ "$both_ckpt_files_exist" = true ]; then
        echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
    else
        echo "Running experiment ${experiment_name}..." | tee -a "$log_file"
        CUDA_VISIBLE_DEVICES=3 python train.py \
            -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
            --gpus 1 \
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
            -t "random_rotation" "normalize" \
            -tp "random_rotation_${rr}" \
            --seed "$seed" | tee -a "$log_file"

        # Check if the process has started successfully
        if [ $? -ne 0 ]; then
            echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
            exit 1
        fi

        # Introduce a short sleep to avoid overloading I/O
        sleep 1
    fi
done

echo "All experiments completed." | tee -a "$log_file"