import os
import torch
import pandas as pd
from pathlib import Path
from torchvision.io import read_video
from torchvision.transforms.v2 import Resize
from pytorchvideo.transforms import UniformTemporalSubsample
from tqdm import tqdm

def main():
    
    csv_path = Path('./dataset_intersections/matched_labels.csv')
    sample_frames = 32
    output_dir = '../test_dataset_tensors_32'
    df = pd.read_csv(csv_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        video_path = row['path']
        video_path = video_path[2:-2]
                        
        label = row['name']
        dictionary = row['dictionary']
        
        tensor_path = Path(output_dir) / label
        
        if not tensor_path.exists():
            os.makedirs(tensor_path)
        
        video_filename = Path(video_path).stem + '.pt'
        video_filename = f'{dictionary}_{video_filename}' 
        
        tensor_full_path = tensor_path / video_filename
        
        frames_data = read_video(video_path)[0]
                
        if frames_data.shape[0] == 0:
            print(f'Error reading video {video_path}')
            continue
        
        frames_data = frames_data.permute(3, 0, 1, 2)
        
        frames_data = Resize((224, 224))(frames_data)
        
        frames_data = UniformTemporalSubsample(sample_frames)(frames_data)

        torch.save(frames_data, tensor_full_path) 

    print(f'Saved all tensors to {output_dir}')

if __name__ == '__main__':
    main()
