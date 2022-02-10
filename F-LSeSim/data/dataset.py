import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

test_transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def remove_file(files, file_name):
    try:
        files.remove(file_name)
    except Exception:
        pass

class XInferenceDataset(Dataset):
    def __init__(self, root_X, transform=None, return_anchor=False, thumbnail=None):
        self.root_X = root_X
        self.transform = transform
        self.return_anchor = return_anchor
        self.thumbnail = thumbnail

        self.X_images = os.listdir(root_X)

        remove_file(self.X_images, "thumbnail.png")
        remove_file(self.X_images, "blank_patches_list.csv")
            
        if self.return_anchor:
            self.__get_boundary()

        self.length_dataset = len(self.X_images)
    
    def __get_boundary(self):
        self.y_anchor_num = 0
        self.x_anchor_num = 0
        for X_image in self.X_images:
            y_idx, x_idx, _, _ = Path(X_image).stem.split("_")[:4]
            y_idx = int(y_idx)
            x_idx = int(x_idx)
            self.y_anchor_num = max(self.y_anchor_num, y_idx)
            self.x_anchor_num = max(self.x_anchor_num, x_idx)
    
    def get_boundary(self):
        assert self.return_anchor == True
        return (self.y_anchor_num, self.x_anchor_num)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        X_img_name = self.X_images[index]

        X_path = os.path.join(self.root_X, X_img_name)

        X_img = np.array(Image.open(X_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img)
            X_img = augmentations["image"]
        
        if self.return_anchor:
            y_idx, x_idx, y_anchor, x_anchor = Path(X_img_name).stem.split("_")[:4]
            y_idx = int(y_idx)
            x_idx = int(x_idx)
            return {"X_img": X_img, "X_path":X_path, "y_idx":y_idx, "x_idx":x_idx, "y_anchor":y_anchor, "x_anchor":x_anchor}
        
        else:
            return {"X_img": X_img, "X_path":X_path}

    def get_thumbnail(self):
        thumbnail_img = np.array(Image.open(self.thumbnail).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=thumbnail_img)
            thumbnail_img = augmentations["image"]
        return thumbnail_img.unsqueeze(0)
