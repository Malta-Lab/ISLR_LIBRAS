import os
import torch
from pathlib import Path
from torchvision.io import read_video
from torchvision.transforms import Resize
from pytorchvideo.transforms import UniformTemporalSubsample
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from tqdm import tqdm

def process_video(video_path, tensor_path, args):
    """Function to process a single video file and save it as a tensor."""
    # Read the video and convert it to tensor
    frames = read_video(video_path)[0]  # Only take video frames, ignore audio
    frames = frames.permute(3, 0, 1, 2)  # Rearrange to (C, T, H, W)
    frames = Resize((frames.shape[-2] // 2, frames.shape[-1] // 2))(frames)  # Resize to half the original size
    
    if not args.not_sample:
        frames = UniformTemporalSubsample(args.frames)(frames)  # Subsample the frames
    
    torch.save(frames, tensor_path)  # Save the tensor
    print(f'Saved tensor to {tensor_path}')

if __name__ == '__main__':
    # Set up argument parser
    parser = ArgumentParser(description="Convert video files to tensors and save them.")
    parser.add_argument('-w', '--workers', type=int, default=8, help="Number of parallel workers.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument('--output_dir', type=str, default=None, help="Path to the directory to save tensors. Defaults to the data_dir name with '_tensors' suffix.")
    parser.add_argument('--frames', type=int, default=16, help="Number of frames to sample from each video.")
    parser.add_argument('--not_sample', action='store_true', help="If set, do not subsample frames, keep all frames.")
    parser.add_argument('--extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mkv', '.wmv', '.mpg', '.mpeg'], help="List of video file extensions to process.")


    args = parser.parse_args()

    # Set up directories
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir.parent / f"{data_dir.name}_tensors_{args.frames if not args.not_sample else 'all_frames'}"

    print(f"Data directory: {data_dir}")
    print(f"Save path: {output_dir}")

    video_paths = []
    tensor_paths = []

    # Collect all video and corresponding tensor paths
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in args.extensions):  # Check if file has a valid video extension
                video_path = Path(root) / file
                relative_path = video_path.relative_to(data_dir)
                tensor_path = output_dir / relative_path.with_suffix('.pt')  # Change extension to .pt
                
                # Ensure the parent directory exists
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                
                video_paths.append(video_path)
                tensor_paths.append(tensor_path)

    # Use ThreadPoolExecutor to process videos in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_video, video_paths, tensor_paths, [args]*len(video_paths)), total=len(video_paths)))
