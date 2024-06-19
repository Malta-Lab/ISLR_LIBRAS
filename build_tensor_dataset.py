import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_video
from torchvision.transforms import Resize
from argparse import ArgumentParser
from pytorchvideo.transforms import UniformTemporalSubsample

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MINDS/')
    parser.add_argument('-n', type=str, default='Acontecer')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--not_sample', action='store_true')

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.frames == 16 and not args.not_sample:
        save_path = Path(f'../MINDS_tensors')
    elif args.not_sample:
        save_path = Path(f'../MINDS_tensors_all_frames')
    else:
        save_path = Path(f'../MINDS_tensors_{args.frames}')

    for root, dirs, files in tqdm(os.walk(data_dir)):
        for file in tqdm(files):
            if file.endswith('.mp4') and args.n in file:  # Adjust if videos have different extensions
                video_path = os.path.join(root, file)

                # Determine the corresponding tensor path
                relative_path = os.path.relpath(video_path, data_dir)
                tensor_path = save_path / relative_path
                tensor_path = str(tensor_path)[:-4] + '.pt'  # Change extension to .pt

                # Ensure the parent directory exists
                Path(tensor_path).parent.mkdir(parents=True, exist_ok=True)

                # Read the video and save as tensor
                frames = read_video(video_path)[0]
                
                frames = frames.permute(3, 0, 1, 2)
                frames = Resize((frames.shape[-2]//2, frames.shape[-1]//2))(frames)
                
                if not args.not_sample:
                    frames = UniformTemporalSubsample(args.frames)(frames)
                
                torch.save(frames, tensor_path)

                print(f'Saved tensor to {tensor_path}')