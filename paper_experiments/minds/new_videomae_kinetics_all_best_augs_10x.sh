#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with specific augmentations

SEEDS_FILE="./seeds.txt"

# Define specific augmentation values
mu="0.25"
cj="0.5"
rp="0.5"
rr="40"

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    # Experiment name includes all the augmentations
    experiment_name="new_videomae_kinetics_all_best_augs_seed_${seed}"
    log_file="./logs/${experiment_name}.log"

    # Run the experiment
    echo "Running experiment ${experiment_name}..." | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 4 \
        -w 16 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "/mnt/G-SSD/BRACIS/MINDS_tensors_32" \
        --exp_name "$experiment_name" \
        --frames 16 \
        --random_sample \
        --dataset "minds" \
        -t "normalize" "color_jitter" "random_perspective" "random_rotation" \
        -tp "color_jitter_${cj}_${cj}_${cj}_${cj}" "random_perspective_${rp}" "random_rotation_${rr}" \
        --mixup \
        --mixup_alpha "$mu" \
        --seed "$seed" | tee -a "$log_file"

    # Check if the process has started successfully
    if [ $? -ne 0 ]; then
        echo "Failed to start the experiment ${experiment_name}." | tee -a "$log_file"
        exit 1
    fi

    # Introduce a short sleep to avoid overloading I/O
    sleep 2

    ((i++))
    # Stop after 100 runs (if needed, although you're only using one set of values here)
    if [ $i -gt 40 ]; then
        break
    fi

done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
