import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from scipy import signal
from yaml.loader import SafeLoader


def gkern(kernlen=1, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def read_yaml_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def reverse_image_normalize(img, mean=0.5, std=0.5):
    return img * std + mean

def reduce_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size) * patch_size

def extend_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size + 1) * patch_size

def get_kernel(padding=1, gaussian_std=3, mode="constant"):
    kernel_size = padding * 2 + 1
    if mode == "constant":
        kernel = torch.ones(1,1,kernel_size,kernel_size)
        kernel = kernel / (kernel_size * kernel_size)

    elif mode == "gaussian":
        kernel = gkern(kernel_size, std=gaussian_std)
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel)
    
    else:
        raise NotImplementedError
    
    return kernel
