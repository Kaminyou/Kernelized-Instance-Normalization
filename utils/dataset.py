import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class XYDataset(Dataset):
    def __init__(self, root_X, root_Y, paired=False, transform=None):
        self.root_X = root_X
        self.root_Y = root_Y
        self.paired = paired
        self.transform = transform

        self.X_images = os.listdir(root_X)
        self.Y_images = os.listdir(root_Y)
        if paired:
            assert len(self.X_images) == len(self.Y_images)
            self.X_images = sorted(self.X_images)
            self.Y_images = sorted(self.Y_images)
            self.length_dataset = len(self.X_images)

        else:
            self.length_dataset = max(len(self.X_images), len(self.Y_images))
            self.X_len = len(self.X_images)
            self.Y_len = len(self.Y_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        if self.paired:
            X_img = self.X_images[index % self.length_dataset]
            Y_img = self.Y_images[index % self.length_dataset]

        else:
            X_img = self.X_images[index % self.X_len]
            random_y_index = random.randint(0, self.Y_len)
            Y_img = self.Y_images[random_y_index % self.Y_len]

        X_path = os.path.join(self.root_X, X_img)
        Y_path = os.path.join(self.root_Y, Y_img)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        Y_img = np.array(Image.open(Y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img, image0=Y_img)
            X_img = augmentations["image"]
            Y_img = augmentations["image0"]

        return X_img, Y_img

class XInferenceDataset(Dataset):
    def __init__(self, root_X, transform=None, return_anchor=False, thumbnail=None):
        self.root_X = root_X
        self.transform = transform
        self.return_anchor = return_anchor
        self.thumbnail = thumbnail

        self.X_images = os.listdir(root_X)
        try:
            self.X_images.remove("thumbnail.png")
        except Exception:
            pass
            
        if self.return_anchor:
            self.__get_boundary()

        self.length_dataset = len(self.X_images)
    
    def __get_boundary(self):
        self.y_anchor_num = 0
        self.x_anchor_num = 0
        for X_image in self.X_images:
            y_idx, x_idx, _, _ = Path(X_image).stem.split("_")
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
            return (X_img, X_path, y_idx, x_idx, y_anchor, x_anchor)
        
        else:
            return (X_img, X_path)

    def get_thumbnail(self):
        thumbnail_img = np.array(Image.open(self.thumbnail).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=thumbnail_img)
            thumbnail_img = augmentations["image"]
        return thumbnail_img.unsqueeze(0)

if __name__ == "__main__":
    from util import test_transforms
    dataset = XInferenceDataset(root_X="./", transform=test_transforms, thumbnail="./data/my_photo/IMG_7867.jpg")
    img = dataset.get_thumbnail()
    print(img.shape)
