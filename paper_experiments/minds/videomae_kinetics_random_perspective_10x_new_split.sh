#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with 10 random perspective

SEEDS_FILE="./seeds.txt"

# Define array of random perspective values (degrees)
random_perspective=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    for rp in "${random_perspective[@]}"
    do
        # Check if both best.ckpt and last.ckpt files exist in the version_X/checkpoints directories
        both_ckpt_files_exist=false
        for dir in /mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/minds/augs/new_split_random_perspective/new_splits_videomae_kinetics_random_perspective_"${rp}"_"${seed}"/version_*/checkpoints; do
            if [ -f "$dir/best_model.ckpt" ] && [ -f "$dir/top5_best_model.ckpt" ]; then
                both_ckpt_files_exist=true
                break
            fi
        done

        experiment_name="new_splits_videomae_kinetics_random_perspective_${rp}_${seed}"
        log_file="./logs/${experiment_name}.log"
        
        if [ "$both_ckpt_files_exist" = true ]; then
            echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
        else
            echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
            CUDA_VISIBLE_DEVICES=3 python train.py \
                -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
                --gpus 1 \
                -sched "plateau" \
                -epochs 900 \
                -w 16 \
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
        fi
        ((i++))
        # 100 runs (10 seeds x 10 random perspective)
        if [ $i -gt 100 ]; then
            break 2 # Breaks out of both loops            
        fi
    done
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
