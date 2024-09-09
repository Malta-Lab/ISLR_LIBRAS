import os
import random
import csv
from sklearn.model_selection import train_test_split

# Set a seed for reproducibility
random.seed(42)

# Parameters
dataset_dir = '../MINDS_tensors_32'  # Replace with the path to your dataset
output_csv = 'minds_dataset_annotations.csv'  # The output CSV file
test_size = 0.25  # 25% of the data goes to the test set

# Initialize a list to store information about each file
data = []

# Traverse the dataset directory
for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    
    # Ensure the directory is actually a label directory
    if os.path.isdir(label_dir):
        files = [f for f in os.listdir(label_dir) if f.endswith('.pt')]
        
        # Parse the file names and append information to the data list
        for file_name in files:
            # File format: 01AcontecerSinalizador01-1.pt
            parts = file_name.split('-')  # Split by '-'
            first_part = parts[0]  # "01AcontecerSinalizador01"
            repetition = parts[1].split('.')[0]  # "1" from "1.pt"
            
            label_id = first_part[:2]  # "01"
            actor_id = first_part[10:]  # "Sinalizador01"
            
            file_path = os.path.join(label_dir, file_name)  # Full path to the tensor file
            
            # Append the data tuple: (label, path, actor_id, repetition)
            data.append((label, file_path, actor_id, repetition))

# Convert the data list into a dictionary where the keys are (label, actor_id) pairs
data_by_label_actor = {}
for label, file_path, actor_id, repetition in data:
    key = (label, actor_id)
    if key not in data_by_label_actor:
        data_by_label_actor[key] = []
    data_by_label_actor[key].append((file_path, repetition))

# Ensure all labels and actors have enough instances and split the data into train/test sets
train_data = []
test_data = []

for (label, actor_id), file_paths_repetitions in data_by_label_actor.items():
    # Ensure there are enough instances for both train and test sets
    if len(file_paths_repetitions) < 2:
        print(f"Skipping label {label} for actor {actor_id} due to insufficient instances.")
        continue
    
    # Sort the instances by repetition number (just to ensure consistency)
    file_paths_repetitions.sort(key=lambda x: int(x[1]))  # Sort by repetition number
    
    # Split the data into train/test (75/25 split)
    file_paths, repetitions = zip(*file_paths_repetitions)
    train_paths, test_paths = train_test_split(file_paths, test_size=test_size, random_state=42)
    
    # Ensure that both train and test sets contain at least 1 instance per label/actor
    if len(train_paths) == 0 or len(test_paths) == 0:
        print(f"Not enough data for label {label} and actor {actor_id}. Adjusting the split.")
        test_paths = [train_paths.pop()]  # Move 1 item to the test set if necessary
    
    # Append to the respective lists with the split designation
    train_data.extend([(label, path, 'train') for path in train_paths])
    test_data.extend([(label, path, 'test') for path in test_paths])

# Combine train and test data
all_data = train_data + test_data

# Write the data to a CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['label', 'path', 'split'])  # Write the header
    
    for row in all_data:
        writer.writerow(row)

print(f"CSV file has been created: {output_csv}")
