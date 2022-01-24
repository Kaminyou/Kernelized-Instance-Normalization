import yaml
from yaml.loader import SafeLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
