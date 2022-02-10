import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import signal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from yaml.loader import SafeLoader

from data import create_dataset
from data.dataset import XInferenceDataset, test_transforms
from models import create_model
from options.train_options import TrainOptions


def reverse_image_normalize(img, mean=0.5, std=0.5):
    return img * std + mean

def read_yaml_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def gkern(kernlen=1, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

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
if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    config = read_yaml_config(opt.config)
    
    model = create_model(opt, normalization_mode=config["INFERENCE_SETTING"]["NORMALIZATION"])      # create a model given opt.model and other options
    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    if config["INFERENCE_SETTING"]["NORMALIZATION"] == "tin":
        test_dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"], 
            transform=test_transforms, 
            return_anchor=True, 
            thumbnail=config["INFERENCE_SETTING"]["THUMBNAIL"]
            )
    else:
        test_dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"], 
            transform=test_transforms, 
            return_anchor=True
            )
            
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    save_path_base = os.path.join(
        config["EXPERIMENT_ROOT_PATH"], 
        config["EXPERIMENT_NAME"], 
        "test", 
        config["INFERENCE_SETTING"]["NORMALIZATION"], 
        config["INFERENCE_SETTING"]["MODEL_VERSION"]
    )
    os.makedirs(save_path_base, exist_ok=True)

    if config["INFERENCE_SETTING"]["NORMALIZATION"] == "tin":
        model.init_thumbnail_instance_norm_for_whole_model()
        thumbnail = test_dataset.get_thumbnail()
        thumbnail_fake = model.inference(thumbnail)
        save_image(
            reverse_image_normalize(thumbnail_fake), 
            os.path.join(save_path_base, "thumbnail_Y_fake.png")
        )

        model.use_thumbnail_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path = data["X_img"], data["X_path"]
            Y_fake = model.inference(X)
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X), 
                    os.path.join(save_path_base, f"{Path(X_path[0]).stem}_X_{idx}.png")
                )
            save_image(
                reverse_image_normalize(Y_fake), 
                os.path.join(save_path_base, f"{Path(X_path[0]).stem}_Y_fake_{idx}.png")
            )    
        
    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "kin":
        save_path_base_kin = os.path.join(save_path_base, f"{config['INFERENCE_SETTING']['KIN_KERNEL']}_{config['INFERENCE_SETTING']['KIN_PADDING']}")
        os.makedirs(save_path_base_kin, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N, anchor_num = N but it actually has N + 1 values
        model.init_kernelized_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1, 
            x_anchor_num=x_anchor_num + 1, 
            kernel=get_kernel(padding=config["INFERENCE_SETTING"]["KIN_PADDING"], mode=config["INFERENCE_SETTING"]["KIN_KERNEL"])
        )
        for idx, data in enumerate(test_loader):
            print(f"Caching {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = data["X_img"], data["X_path"], data["y_idx"], data["x_idx"]
            _ = model.inference_with_anchor(X, y_anchor=y_anchor, x_anchor=x_anchor, padding=config["INFERENCE_SETTING"]["KIN_PADDING"])

        model.use_kernelized_instance_norm_for_whole_model(padding=config["INFERENCE_SETTING"]["KIN_PADDING"])
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = data["X_img"], data["X_path"], data["y_idx"], data["x_idx"]
            Y_fake = model.inference_with_anchor(X, y_anchor=y_anchor, x_anchor=x_anchor, padding=config["INFERENCE_SETTING"]["KIN_PADDING"])
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X), 
                    os.path.join(save_path_base_kin, f"{Path(X_path[0]).stem}_X_{idx}.png")
                )
            save_image(
                reverse_image_normalize(Y_fake), 
                os.path.join(save_path_base_kin, f"{Path(X_path[0]).stem}_Y_fake_{idx}.png")
            )    

    else:
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")

            X, X_path = data["X_img"], data["X_path"]
            Y_fake = model.inference(X)

            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X), 
                    os.path.join(save_path_base, f"{Path(X_path[0]).stem}_X_{idx}.png")
                )
            
            save_image(
                reverse_image_normalize(Y_fake), 
                os.path.join(save_path_base, f"{Path(X_path[0]).stem}_Y_fake_{idx}.png")
            )
