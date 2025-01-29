#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics with mixup values

# Define array of mixup alpha values
mixup=("0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4" "4.5" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Ensure log directory exists
mkdir -p ./logs

# Fixed seed value
seed=42  # Single seed as in your first script

for mu in "${mixup[@]}"
do
    # Check if both best.ckpt and last.ckpt files exist in the version_X/checkpoints directories
    both_ckpt_files_exist=false
    for dir in /mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/agressive_splits_videomae_kinetics_mixup_"${mu}"_"${seed}"/version_*/checkpoints; do
        if [ -f "$dir/best_model.ckpt" ] && [ -f "$dir/top5_best_model.ckpt" ]; then
            both_ckpt_files_exist=true
            break
        fi
    done

    experiment_name="agressive_splits_videomae_kinetics_mixup_${mu}_${seed}"
    log_file="./logs/${experiment_name}.log"
    
    if [ "$both_ckpt_files_exist" = true ]; then
        echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
    else
        echo "Starting experiment ${experiment_name}..." | tee -a "$log_file"
        CUDA_VISIBLE_DEVICES=4 python train.py \
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
            -t "normalize" \
            --mixup \
            --mixup_alpha "$mu" \
            --seed "$seed" | tee -a "$log_file"

        # Check if the process failed
        if [ $? -ne 0 ]; then
            echo "Failed to complete experiment ${experiment_name}." | tee -a "$log_file"
            exit 1
        fi

        # Short sleep to avoid I/O overload
        sleep 1
    fi
done

echo "All experiments completed." | tee -a "$log_file"