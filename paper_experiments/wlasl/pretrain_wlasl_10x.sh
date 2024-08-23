#!/bin/bash
# videomae on WLASL dataset with pre-train on kinetics and no augmentations 10x (1 per seed)

SEEDS_FILE="./seeds.txt"

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    # Define the experiment name and log file
    experiment_name="WLASL_pretrain_seed_$seed"
    log_file="./logs/${experiment_name}.log"

    # Check if both best.ckpt and last.ckpt files exist in the version_X/checkpoints directories
    both_ckpt_files_exist=false
    for dir in /mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/wlasl/WLASL_pretrain_seed_"${seed}"/version_*/checkpoints; do
        if [ -f "$dir/best_model.ckpt" ] && [ -f "$dir/last.ckpt" ]; then
            both_ckpt_files_exist=true
            break
        fi
    done

    if [ "$both_ckpt_files_exist" = true ]; then
        echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
    else
        echo "Running ${experiment_name} $i with seed $seed" | tee -a "$log_file"

        # Run the training script
        CUDA_VISIBLE_DEVICES=4,5 python train_v3.py \
            -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
            --gpus 2 \
            -sched "plateau" \
            -lr 0.0001 \
            --data_path "/mnt/G-SSD/BRACIS/WLASL_tensors_32" \
            --exp_name "$experiment_name" \
            --frames 16 \
            --random_sample \
            --dataset "wlasl" \
            -t "normalize" \
            --seed "$seed" | tee -a "$log_file"

        # Check if the process has started successfully
        if [ $? -ne 0 ]; then
            echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
            exit 1
        fi

        # Introduce a short sleep to avoid overloading I/O
        sleep 2
    fi

    ((i++))

    # Exit loop after 10 runs
    if [ $i -gt 10 ]; then
        break
    fi
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
