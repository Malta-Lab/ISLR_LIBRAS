from torchvision.transforms.v2 import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomCrop,
    ColorJitter,
    GaussianBlur,
    RandomRotation,
    Normalize,
    RandomPerspective,
    UniformTemporalSubsample,
    AugMix,
)
import torch
from utils import preprocess_parameters


class RandomTemporalSubsample:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, video):
        if video.shape[0] < self.num_frames:
            return video

        indices = torch.randint(0, video.shape[0], (self.num_frames,))
        indices = torch.sort(indices).values
        return video[indices]

    def __repr__(self):
        return self.__class__.__name__ + f"(num_frames={self.num_frames})"
    
class Permute:
    def __init__(self, order):
        self.order = order

    def __call__(self, video):
        return video.permute(self.order)

    def __repr__(self):
        return self.__class__.__name__ + f"(order={self.order})"
    
class ToFloat():
    def __call__(self, video):
        return video.float()

    def __repr__(self):
        return self.__class__.__name__ + f"()"

class Transforms:
    def __init__(
        self,
        transforms_list,
        resize_dims=(224, 224),
        sample_frames=16,
        random_sample=False,
        dataset_name="minds",
        transforms_parameters=None,
    ):
        self.available_transforms = [
            "color_jitter",
            "gaussian_blur",
            "normalize",
            "random_horizontal_flip",
            "random_perspective",
            "random_rotation",
        ]

        
        # check transforms
        for transform in transforms_list:
            if transform not in self.available_transforms:
                raise ValueError(f"Transform {transform} not available")
                    
        self.transforms_list = transforms_list
        self.resize_dims = resize_dims
        self.sample_frames = sample_frames
        self.random_sample = random_sample
        self.dataset_name = dataset_name
        self.transforms_parameters = transforms_parameters
        self.matcher = self.__match_transforms_and_parameters(transforms_list, transforms_parameters)
        self.transforms = self.__build_transforms()
        

    def __build_transforms(
        self,
    ):
        transforms = []
        
        transforms.append(Resize(self.resize_dims, antialias=True))
        # transforms.append(lambda x: x.permute(1, 0, 2, 3))
        transforms.append(Permute((1, 0, 2, 3)))
        
        # add color jitter
        if "color_jitter" in self.transforms_list:
            transforms.append(ColorJitter(*self.matcher.get("color_jitter", (0.5, 0.5, 0.5, 0.5))))
            
        # transform to float
        transforms.append(ToFloat())
        
        # add sampler
        if self.random_sample:
            transforms.append(RandomTemporalSubsample(self.sample_frames))
        else:
            transforms.append(UniformTemporalSubsample(self.sample_frames))
            
        if "gaussian_blur" in self.transforms_list:
            transforms.append(GaussianBlur(*self.matcher.get("gaussian_blur", (3,(0.1, 2.0)))))
            
        # normalization
        if "normalize" in self.transforms_list:
            transforms.append(self.__normalization())
            
        # orientation transforms
        if "random_horizontal_flip" in self.transforms_list:
            transforms.append(RandomHorizontalFlip(*self.matcher.get("random_horizontal_flip", ())))
        if "random_perspective" in self.transforms_list:
            transforms.append(RandomPerspective(*self.matcher.get("random_perspective", ())))
        if "random_rotation" in self.transforms_list:
            transforms.append(RandomRotation(*self.matcher.get("random_rotation", ())))
            
        return Compose(transforms)
    
    def __match_transforms_and_parameters(self,transforms_list, transforms_parameters):
        matcher = {}
        
        if transforms_parameters is not None:
            for params in transforms_parameters:
                for tname in transforms_list:
                    if tname in params:
                        parameters = preprocess_parameters(params, tname)
                        matcher[tname] = (parameters)
        return matcher
                    
        
    def __normalization(self,):
        if self.dataset_name == "minds":
            return Normalize((118.4939, 118.4997, 118.5007), (57.2457, 57.2454, 57.2461))
        if self.dataset_name == "slovo":
            return Normalize((143.2916, 133.0764, 128.7852), (62.1262, 64.2752, 62.8632))
        if self.dataset_name == "test":
            return Normalize((110.0589, 104.0907, 119.5019), (53.9639, 47.5653, 51.7986))
        if self.dataset_name == "wlasl":
            return Normalize((108.7010, 109.7283, 103.2872), (60.1688, 54.0494, 43.9858))
        
    def __call__(self, video):
        return self.transforms(video)