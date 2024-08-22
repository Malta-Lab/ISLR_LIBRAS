import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def process_tensor(tensor_path):
    # Load the tensor file
    tensor = torch.load(tensor_path)
    tensor = tensor.float()

    # Calculate the sum and sum of squares for the tensor across spatial dimensions (height and width) for each channel
    tensor_sum = tensor.sum(dim=[0, 2, 3]).numpy()  # Sum across frames, height, and width
    tensor_sum_squared = (tensor ** 2).sum(dim=[0, 2, 3]).numpy()  # Sum of squares across frames, height, and width
    num_elements = tensor.size(0) * tensor.size(2) * tensor.size(3)  # Number of elements per channel

    return tensor_sum, tensor_sum_squared, num_elements

def calculate_mean_std(tensor_folder_path, num_workers):
    total_sum = np.zeros(3)
    total_sum_squared = np.zeros(3)
    total_elements = 0

    tensor_files = [os.path.join(tensor_folder_path, f) for f in os.listdir(tensor_folder_path) if f.endswith('.pt')]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_tensor, tensor_file): tensor_file for tensor_file in tensor_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tensors"):
            result = future.result()
            if result:
                tensor_sum, tensor_sum_squared, num_elements = result
                total_sum += tensor_sum
                total_sum_squared += tensor_sum_squared
                total_elements += num_elements

    # Calculate mean and std for each channel
    mean = total_sum / total_elements
    std = np.sqrt(total_sum_squared / total_elements - mean ** 2)

    return tuple(mean), tuple(std)

def save_stats_to_file(mean, std, output_file):
    with open(output_file, 'w') as f:
        f.write(f'Mean: {mean}\n')
        f.write(f'Std: {std}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean and std for each RGB channel from a dataset of .pt tensor files.")
    parser.add_argument("tensor_folder_path", type=str, help="Path to the folder containing .pt tensor files.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker threads.")
    parser.add_argument("--output_file", type=str, default="dataset_mean_std.txt", help="Output file to save the mean and std values.")
    
    args = parser.parse_args()
    
    mean, std = calculate_mean_std(args.tensor_folder_path, args.num_workers)
    
    save_stats_to_file(mean, std, args.output_file)
    
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print(f'Statistics saved to {args.output_file}')
