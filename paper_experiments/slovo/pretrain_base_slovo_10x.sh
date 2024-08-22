#!/bin/bash
# videomae on SLOVO dataset with pre-train on kinetics and no augmentations 10x (1 per seed)

SEEDS_FILE="./seeds.txt"

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    # Define the experiment name and log file
    experiment_name="slovo_pretrain_seed_$seed"
    log_file="./logs/${experiment_name}.log"
    
    echo "Running ${experiment_name} $i with seed $seed" | tee -a "$log_file"

    # Run the training script
    CUDA_VISIBLE_DEVICES=2,3 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 2 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/slovo_tensors_32/slovo_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --random_sample \
        --dataset "slovo" \
        -t "normalize" \
        --seed "$seed" 2>> "$log_file" | tee -a "$log_file"

    # Check if the process has started successfully
    if [ $? -ne 0 ]; then
        echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    # Introduce a short sleep to avoid overloading I/O
    sleep 2

    ((i++))

    # Exit loop after 10 runs
    if [ $i -gt 10 ]; then
        break
    fi
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
