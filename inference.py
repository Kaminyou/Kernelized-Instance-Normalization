import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.cut import ContrastiveModel
from utils.dataset import XInferenceDataset
from utils.util import (read_yaml_config, reverse_image_normalize,
                        test_transforms)


def main():
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    model = ContrastiveModel(config, normalization=config["INFERENCE_SETTING"]["NORMALIZATION"])

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

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    if config["INFERENCE_SETTING"]["NORMALIZATION"] == "tin":
        os.makedirs(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "tin"), exist_ok=True)
        model.init_thumbnail_instance_norm_for_whole_model()
        thumbnail = test_dataset.get_thumbnail()
        thumbnail_fake = model.inference(thumbnail)
        save_image(
            reverse_image_normalize(thumbnail_fake), 
            os.path.join(config["EXPERIMENT_ROOT_PATH"],
            config["EXPERIMENT_NAME"],
            "test",
            "tin", 
            "thumbnail_Y_fake.png")
        )

        model.use_thumbnail_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, _, _, _, _ = data
            Y_fake = model.inference(X)
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X), 
                    os.path.join(config["EXPERIMENT_ROOT_PATH"], 
                    config["EXPERIMENT_NAME"], 
                    "test",
                    "tin", 
                    f"{Path(X_path[0]).stem}_X_{idx}.png")
                )
            save_image(
                reverse_image_normalize(Y_fake), 
                os.path.join(config["EXPERIMENT_ROOT_PATH"], 
                config["EXPERIMENT_NAME"], 
                "test",
                "tin", 
                f"{Path(X_path[0]).stem}_Y_fake_{idx}.png")
            )    
        
    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "kin":
        os.makedirs(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "kin"), exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N, anchor_num = N but it actually has N + 1 values
        model.init_kernelized_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1, 
            x_anchor_num=x_anchor_num + 1, 
            kernel=torch.ones(3,3)
        )
        for idx, data in enumerate(test_loader):
            print(f"Caching {idx}", end="\r")
            X, X_path, y_anchor, x_anchor, _, _ = data
            _ = model.inference_with_anchor(X, y_anchor=y_anchor, x_anchor=x_anchor, padding=config["INFERENCE_SETTING"]["PADDING"])

        model.use_kernelized_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, y_anchor, x_anchor, _, _ = data
            Y_fake = model.inference_with_anchor(X, y_anchor=y_anchor, x_anchor=x_anchor, padding=config["INFERENCE_SETTING"]["PADDING"])
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X), 
                    os.path.join(config["EXPERIMENT_ROOT_PATH"], 
                    config["EXPERIMENT_NAME"], 
                    "test",
                    "kin", 
                    f"{Path(X_path[0]).stem}_X_{idx}.png")
                )
            save_image(
                reverse_image_normalize(Y_fake), 
                os.path.join(config["EXPERIMENT_ROOT_PATH"], 
                config["EXPERIMENT_NAME"], 
                "test",
                "kin", 
                f"{Path(X_path[0]).stem}_Y_fake_{idx}.png")
            )    

    else:
        os.makedirs(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "in"), exist_ok=True)
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")

            X, X_path, _, _, _, _  = data
            Y_fake = model.inference(X)

            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(reverse_image_normalize(X), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "in", f"{Path(X_path[0]).stem}_X_{idx}.png"))
            
            save_image(reverse_image_normalize(Y_fake), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "in", f"{Path(X_path[0]).stem}_Y_fake_{idx}.png"))

if __name__ == "__main__":
    main()
