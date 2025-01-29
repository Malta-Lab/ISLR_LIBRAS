#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with specific augmentations

SEEDS_FILE="./seeds.txt"

# Define specific augmentation values
mu="0.10"
cj="0.4"
rp="0.5"
rr="0.3"

# Ensure log directory exists
mkdir -p ./logs

# Reduced number of data loading workers to minimize CPU load
NUM_WORKERS=16  # Reduced from 16 to 8 (adjust based on your CPU cores)

# Read each line from the seeds file
while IFS= read -r seed
do
    # Experiment name includes all the augmentations
    experiment_name="videomae_kinetics_all_best_F1augs_seed_${seed}_OLD_SPLIT"
    log_file="./logs/${experiment_name}.log"

    # Run the experiment
    echo "Running experiment ${experiment_name}..." | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=2,3 python train_test.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 2 \
        -epochs 900 \
        -w "$NUM_WORKERS" \
        -bs 16 \
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

    # Check the exit status of the last command
    if [ $? -ne 0 ]; then
        echo "Warning: Experiment ${experiment_name} failed. Continuing to next seed." | tee -a "$log_file"
    fi

    # Clean up GPU memory and add short pause between experiments
    sleep 10  # Increased sleep to ensure GPU memory cleanup

done < "$SEEDS_FILE"

echo "All experiments completed." | tee -a "$log_file"