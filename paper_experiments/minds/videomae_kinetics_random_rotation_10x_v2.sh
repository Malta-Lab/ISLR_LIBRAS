#!/bin/bash
# videomae on MINDS dataset with pre-train on kinetics 10x (1 per seed) with 10 random rotations

SEEDS_FILE="./seeds.txt"

# Define array of random rotation values (degrees)

random_rotation=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Ensure log directory exists
mkdir -p ./logs

# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    for rr in "${random_rotation[@]}"
    do
        
        csv_exists=false
        experiment_folder="/mnt/G-SSD/BRACIS/BRACIS-2024/lightning_logs/minds/augs/random_rotation/videomae_kinetics_random_rotation_${rr}_${seed}"
        if [ -f "${experiment_folder}/test_metrics_results.csv" ]; then
            csv_exists=true
        fi

        experiment_name="videomae_kinetics_random_rotation_${rr}_${seed}"
        log_file="./logs/${experiment_name}.log"
        
        if [ "$csv_exists" = true ]; then
            echo "Experiment ${experiment_name} already executed. Skipping..." | tee -a "$log_file"
        else
            echo "Experiment ${experiment_name} not executed. Running..." | tee -a "$log_file"
            CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train_v2.py \
                -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
                --gpus 5 \
                -w 16 \
                -epochs 1200 \
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
        ((i++))
        # 100 runs (10 seeds x 10 random rotation)
        if [ $i -gt 100 ]; then
            break 2 # Breaks out of both loops            
        fi
    done
done < "$SEEDS_FILE"

echo "All specified experiments completed." | tee -a "$log_file"
