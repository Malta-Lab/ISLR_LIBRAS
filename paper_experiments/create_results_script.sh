#!/bin/bash
# videomae on MINDS dataset - check all experiments for results and process missing ones

# Base folder containing the experiment folders
BASE_FOLDER="./lightning_logs/minds/augs/random_rotation"

# Loop over all experiment folders matching the pattern
for experiment_folder in "${BASE_FOLDER}"/videomae_kinetics_random_rotation_*
do
    # Define the paths for the CSV and the checkpoint file
    csv_file="${experiment_folder}/test_metrics_results.csv"
    experiment_name=$(basename "$experiment_folder")

    # Check if the CSV file exists
    if [ -f "$csv_file" ]; then
        echo "Experiment ${experiment_name} already has results (CSV exists). Skipping..." 
    else
        echo "Experiment ${experiment_name} missing results (CSV not found). Processing..." 

        # Find the .ckpt file in the version_*/checkpoints directories
        ckpt_file=""
        for dir in "${experiment_folder}"/version_*/checkpoints; do
            if [ -f "$dir/best_model.ckpt" ]; then
                ckpt_file="$dir/best_model.ckpt"
                break
            fi
        done

        if [ -z "$ckpt_file" ]; then
            echo "No checkpoint file found for experiment ${experiment_name}. Skipping..." 
            continue
        fi

        echo "Found checkpoint file: $ckpt_file. Running results script..." 

        # Run the create_results_v2.py script with the .ckpt file
        CUDA_VISIBLE_DEVICES=0 python create_results_v2.py -dm "test" --base_dir "${experiment_folder}" -cuda 2 -w 16

        # Check if the process has started successfully
        if [ $? -ne 0 ]; then
            echo "Failed to process results for experiment ${experiment_name}." 
            continue # Skip this experiment and move to the next one
        fi
    fi
done

echo "All experiments processed."
