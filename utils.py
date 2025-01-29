import os
import random
import torch
import numpy as np
from torchvision.io import read_video
from torchvision.transforms.v2 import Resize, Normalize, UniformTemporalSubsample


def read_single_video(video_as_mp4, device):
    frames = read_video(video_as_mp4)[0].float()
    frames = frames.permute(3, 0, 1, 2)
    frames = frames.permute(1, 0, 2, 3)
    frames = Resize((224, 224))(frames)
    frames = Normalize((118.4939, 118.4997, 118.5007), (57.2457, 57.2454, 57.2461))(
        frames
    )
    frames = UniformTemporalSubsample(16)(frames)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)
    return frames
    
def create_labels2idx(dataset):
    label2idx = []

    for (video, label) in dataset._labeled_videos:
        label_text = video.split('/')[2]
        label2idx.append((label_text, label['label']))
        
    return label2idx

def preprocess_parameters(x, name):
    x = x.replace(name, '')
    x = x.split('_')
    parameters = x[1:]
    parameters = [float(p) for p in parameters]
    return parameters

def set_seed(seed=42):

    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

CLASSES2IDX = {
    'acontecer': 0, 'aluno': 1, 'amarelo': 2, 'america': 3, 'aproveitar': 4, 'bala': 5, 'banco': 6, 'banheiro': 7, 'barulho': 8, 'cinco': 9, 'conhecer': 10, 'espelho': 11, 'esquina': 12, 'filho': 13, 'maca': 14, 'medo': 15, 'ruim': 16, 'sapo': 17, 'vacina': 18, 'vontade': 19
}