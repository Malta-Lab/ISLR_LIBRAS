import torch
from torch import nn
import kornia.augmentation as K
from typing import List, Tuple, Dict, Optional

class VideoTransforms(nn.Module):
    def __init__(
        self,
        mode: str = 'train',
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        dataset_name: str = "minds",
        transforms_list: List[str] = None,
        transforms_parameters: List[str] = None,
        use_kinetics_norm: bool = False
    ):
        super().__init__()
        self.mode = mode
        self.num_frames = num_frames
        self.input_size = input_size
        
        # Parse parameters from strings like "color_jitter_0.5_0.5_0.5_0.5"
        self.params = self._parse_parameters(transforms_parameters or [])
        self.norm_mean, self.norm_std = self._get_normalization_params(dataset_name, use_kinetics_norm)
        self.temporal_sampler = self._get_temporal_sampler()
        self.spatial_transforms = self._build_spatial_transforms(transforms_list or [])

    def _parse_parameters(self, params: List[str]) -> Dict[str, List[float]]:
        parsed = {}
        for param_str in params:
            # Find the first occurrence of a digit
            idx = None
            for i, ch in enumerate(param_str):
                if ch.isdigit():
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"No numeric value found in parameter: {param_str}")
            # The name is everything before the first digit (strip any trailing underscore)
            name = param_str[:idx].rstrip('_')
            values_str = param_str[idx:]
            try:
                values = [float(v) for v in values_str.split('_') if v]
            except ValueError as e:
                raise ValueError(f"Invalid float conversion in parameter: {param_str}") from e
            parsed[name] = values
        return parsed

    def _get_normalization_params(self, dataset_name: str, use_kinetics: bool) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        norms = {
            "minds": ((118.49, 118.50, 118.50), (57.25, 57.25, 57.25)),
            "slovo": ((143.29, 133.08, 128.79), (62.13, 64.28, 62.86)),
            "wlasl": ((108.70, 109.73, 103.29), (60.17, 54.05, 43.99)),
            "kinetics": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        }
        if use_kinetics:
            return norms["kinetics"]
        else:
            params = norms.get(dataset_name, norms["kinetics"])
            mean, std = params
            # Convert parameters from the [0,255] scale to [0,1]
            return (tuple(m / 255.0 for m in mean), tuple(s / 255.0 for s in std))

    def _get_temporal_sampler(self):
        return self._random_temporal_subsample if self.mode == 'train' else self._uniform_temporal_subsample

    def _build_spatial_transforms(self, transforms_list: List[str]) -> nn.Sequential:
        transforms = []
        
        if 'color_jitter' in transforms_list:
            params = self.params.get('color_jitter', [0.5, 0.5, 0.5, 0.5])
            transforms.append(K.ColorJitter(*params, p=1.0))

        if 'random_perspective' in transforms_list:
            distortion = self.params.get('random_perspective', [0.5])[0]
            transforms.append(K.RandomPerspective(distortion_scale=distortion, p=1.0))

        if 'random_rotation' in transforms_list:
            degrees = self.params.get('random_rotation', [30])[0]
            transforms.append(K.RandomRotation(degrees=degrees, p=1.0))

        transforms.append(K.Resize(self.input_size, antialias=True))
        return nn.Sequential(*transforms)

    def _random_temporal_subsample(self, video: torch.Tensor) -> torch.Tensor:
        # Input shape: (C, T, H, W)
        T = video.size(1)
        if T < self.num_frames:
            return video
        start = torch.randint(0, T - self.num_frames + 1, (1,)).item()
        return video[:, start:start+self.num_frames]

    def _uniform_temporal_subsample(self, video: torch.Tensor) -> torch.Tensor:
        # Input shape: (C, T, H, W)
        T = video.size(1)
        if T <= self.num_frames:
            return video
        indices = torch.linspace(0, T-1, self.num_frames).long()
        return video[:, indices]

    def _apply_normalization(self, video: torch.Tensor) -> torch.Tensor:
        # Manual normalization for video tensors (C, T, H, W)
        mean = torch.tensor(self.norm_mean, device=video.device).view(-1, 1, 1, 1)
        std = torch.tensor(self.norm_std, device=video.device).view(-1, 1, 1, 1)
        return (video - mean) / std

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # Input shape: (C, T, H, W)
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
            
        # Temporal sampling
        video = self.temporal_sampler(video)
        
        # Spatial transforms (applied per frame)
        C, T, H, W = video.shape
        video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
        transformed_frames = []
        for frame in video:
            # Add batch dimension for Kornia: (1, C, H, W)
            frame = frame.unsqueeze(0)
            frame = self.spatial_transforms(frame)
            frame = frame.squeeze(0)
            transformed_frames.append(frame)
        video = torch.stack(transformed_frames)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # Back to (C, T, H, W)
        
        # Apply normalization
        return self._apply_normalization(video)
