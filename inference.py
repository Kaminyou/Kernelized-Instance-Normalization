import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.model import get_model
from utils.dataset import XInferenceDataset
from utils.util import (get_kernel, read_yaml_config, reverse_image_normalize,
                        test_transforms)


def main():
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument("-c", "--config", type=str, default="./data/example/config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    model = get_model(config=config, model_name=config["MODEL_NAME"], normalization=config["INFERENCE_SETTING"]["NORMALIZATION"])

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

if __name__ == "__main__":
    main()
