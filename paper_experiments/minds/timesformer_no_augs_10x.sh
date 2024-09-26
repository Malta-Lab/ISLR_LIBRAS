#!/bin/bash
# timesformer on MINDS dataset, no pre-train on kinetics, no augs, with 10 seeds from seeds.txt

SEEDS_FILE="./seeds.txt"

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    # Check if both best.ckpt and last.ckpt files exist in the version_X/checkpoints directories
    both_ckpt_files_exist=false
    for dir in /mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/minds/no_pretrain/timesformer_no_augmentation_"${seed}"/version_*/checkpoints; do
        if [ -f "$dir/best_model.ckpt" ] && [ -f "$dir/top5_best_model.ckpt" ]; then
            both_ckpt_files_exist=true
            break
        fi
    done

    experiment_name="timesformer_no_augmentations_${seed}"
    log_file="./logs/${experiment_name}.log"
    
    if [ "$both_ckpt_files_exist" = true ]; then
        echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
    else
        echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
        CUDA_VISIBLE_DEVICES=1,3,4,5 python train.py \
            -ptm "facebook/timesformer-base-finetuned-k400" \
            --gpus 4 \
            --no_pretrain \
            -w 16 \
            -sched "plateau" \
            -lr 0.0001 \
            --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
            --exp_name "$experiment_name" \
            --frames 16 \
            --random_sample \
            --dataset "minds" \
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
    # 10 runs (10 seeds)
    if [ $i -gt 10 ]; then
        break
    fi
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
