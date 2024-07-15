import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None

    frame_means = []
    frame_stds = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame in video file {video_path}")
            break

        # Convert frame from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate mean and std for the frame
        frame_mean = np.mean(frame, axis=(0, 1))
        frame_std = np.std(frame, axis=(0, 1))

        frame_means.append(frame_mean)
        frame_stds.append(frame_std)

    cap.release()

    if frame_means and frame_stds:
        video_mean = np.mean(frame_means, axis=0)
        video_std = np.mean(frame_stds, axis=0)
        return video_mean, video_std
    else:
        print(f"No valid frames found in video file {video_path}")
        return None

def calculate_mean_std(video_folder_path, num_workers):
    means = []
    stds = []

    video_files = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, video_file): video_file for video_file in video_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            if result:
                video_mean, video_std = result
                means.append(video_mean)
                stds.append(video_std)

    if means and stds:
        overall_mean = np.mean(means, axis=0)
        overall_std = np.mean(stds, axis=0)
    else:
        overall_mean = np.array([np.nan, np.nan, np.nan])
        overall_std = np.array([np.nan, np.nan, np.nan])

    return overall_mean, overall_std

def save_stats_to_file(mean, std, output_file):
    with open(output_file, 'w') as f:
        f.write(f'Mean: {mean.tolist()}\n')
        f.write(f'Std: {std.tolist()}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean and std for each RGB channel from a dataset of .mp4 videos.")
    parser.add_argument("video_folder_path", type=str, help="Path to the folder containing .mp4 video files.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker threads.")
    parser.add_argument("--output_file", type=str, default="slovo_dataset_mean_std.txt", help="Output file to save the mean and std values.")
    
    args = parser.parse_args()
    
    mean, std = calculate_mean_std(args.video_folder_path, args.num_workers)
    
    save_stats_to_file(mean, std, args.output_file)
    
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print(f'Statistics saved to {args.output_file}')
