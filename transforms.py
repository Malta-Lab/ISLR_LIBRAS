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
)
import torch

from pytorchvideo.transforms import (
    AugMix)

class RandomTemporalSubsample:
    def __init__(self, num_frames):
        self.num_frames = num_frames
    
    def __call__(self, video):
        # selec random frames form the video, but follow chronological order
        # If i want to retrieve two frames, I would like to get the following possible outputs: [0,1], [0,2], [2,4], [0,3], [1,4]
        
        if video.shape[0] < self.num_frames:
            return video
        indices = torch.randint(0, video.shape[0], (self.num_frames,))
        indices = torch.sort(indices).values
        return video[indices]


MAPPER = {
    "random_crop": RandomCrop,
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_perspective": RandomPerspective,
    'random_rotation': RandomRotation,
}


def build_transforms_v2():
    return Compose(
        [
            Resize((224, 224), antialias=True),
            lambda x: x.permute(1, 0, 2, 3),
        ]
    )


def build_transforms(
    transforms_list,
    resize_dims=(224, 224),
    sample_frames=16,
    random_sample=False,
):
    """ "
    - transforms_list: list of strings with the names of the transforms to be applied
    - resize_dims: tuple with the dimensions of the resized image
    - random_resize: if True, the image will be randomly resized and cropped to the resize_dims
    - sample_frames: if not None, the video will be sampled to this number of frames
    """
    transforms = []
    
    transforms.append(Resize(resize_dims, antialias=True))
        
    # C x T x H x W    ->   T x C x H x W
    transforms.append(lambda x: x.permute(1, 0, 2, 3))
    
    if 'color_jitter' in transforms_list:
        transforms.append(ColorJitter(.5, .5, .5, .5))
        transforms_list.remove('color_jitter')

    transforms.append(lambda x: x.float())

    # sample frames
    if random_sample:
        transforms.append(RandomTemporalSubsample(sample_frames))
    else:
        transforms.append(UniformTemporalSubsample(sample_frames))   
    
    if 'aug_mix' in transforms_list:
        transforms.append(AugMix())
        transforms_list.remove('aug_mix')
        
    if 'gaussian_blur' in transforms_list:
        transforms.append(GaussianBlur(kernel_size=3))
        transforms_list.remove('gaussian_blur')
        
    if "normalize" in transforms_list:
        transforms.append(Normalize((118.4939, 118.4997, 118.5007), (57.2457, 57.2454, 57.2461)))
        transforms_list.remove("normalize")
        
    # add the rest of the transforms
    for transform in transforms_list:
        if transform in MAPPER:
            transforms.append(MAPPER[transform]())

    return Compose(transforms)
