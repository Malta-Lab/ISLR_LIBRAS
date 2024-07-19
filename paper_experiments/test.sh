#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with 10 color jitter

SEEDS_FILE="./seeds.txt"

# Define array of color jitter values
color_jitters=("0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    for cj in "${color_jitters[@]}"
    do
        # Check if any version_X/checkpoints directory exists
        checkpoints_exist=false
        for dir in ./lightning_logs/videomae_kinetics_color_jitter_${cj}_${seed}/version_*/checkpoints; do
            if [ -d "$dir" ]; then
                checkpoints_exist=true
                break
            fi
        done

        if [ "$checkpoints_exist" = true ]; then
            echo "Experiment videomae_kinetics_color_jitter_${cj}_${seed} already executed. Skipping..."
        else
            echo "Experiment videomae_kinetics_color_jitter_${cj}_${seed} not executed. Running..."
            CUDA_VISIBLE_DEVICES=1 python train.py \
                -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
                --gpus 1 \
                -sched "plateau" \
                -lr 0.0001 \
                --data_path "../MINDS_tensors_32" \
                --exp_name "videomae_kinetics_color_jitter_${cj}_${seed}" \
                --frames 16 \
                --random_sample \
                --dataset "minds" \
                -t "color_jitter" "normalize" \
                -tp "color_jitter_${cj}_${cj}_${cj}_${cj}" \
                --seed "$seed"

            # Check if the process has started successfully
            if [ $? -ne 0 ]; then
                echo "Failed to start the experiment videomae_kinetics_color_jitter_${cj}_${seed}."
                exit 1
            fi
        fi
        ((i++))
        # 100 runs (10 seeds x 10 color jitters)
        if [ $i -gt 100 ]; then
            break 2 # Breaks out of both loops            
        fi
    done
done < "$SEEDS_FILE"
