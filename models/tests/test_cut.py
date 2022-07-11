import os
from pathlib import Path

import numpy as np
import pytest
import torch
from models.model import get_model
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from utils.dataset import XInferenceDataset
from utils.util import (read_yaml_config, reverse_image_normalize,
                        test_transforms)

from ..cut import ContrastiveModel


@pytest.fixture()
def config():
    config_path = '/home/vincentwu/URUST_exp/config_lung_lesion_for_paper.yaml'
    config = read_yaml_config(config_path)

    return config

@pytest.fixture()
def model(config):
    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        normalization=config["INFERENCE_SETTING"]["NORMALIZATION"],
        isTrain=False,
    )
    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    return model

@pytest.fixture()
def dataset(config):
    dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
    )
    return dataset

@pytest.fixture()
def dataloader(dataset):
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True
    )
    return loader

@pytest.fixture()
def save_path_base(config):
    save_path_root = os.path.join(
        config["EXPERIMENT_ROOT_PATH"],
        config["EXPERIMENT_NAME"],
        "test"
    )
    save_path_base = os.path.join(
        save_path_root,
        config["INFERENCE_SETTING"]["NORMALIZATION"],
        config["INFERENCE_SETTING"]["MODEL_VERSION"],
    )
    save_path_base_kin = os.path.join(
        save_path_base,
        f"{config['INFERENCE_SETTING']['KIN_KERNEL']}_"
        f"{config['INFERENCE_SETTING']['KIN_PADDING']}",
    )

    return save_path_base_kin

def test_inference_with_anchor(config, model, dataset, dataloader, save_path_base):
    y_anchor_num, x_anchor_num = dataset.get_boundary()

    model.init_kernelized_instance_norm_for_whole_model(
        y_anchor_num=y_anchor_num + 1,
        x_anchor_num=x_anchor_num + 1,
        kernel_padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
        kernel_mode=config["INFERENCE_SETTING"]["KIN_KERNEL"],
    )

    for idx, data in enumerate(dataloader):
        X, X_path, y_anchor, x_anchor = (
            data["X_img"],
            data["X_path"],
            data["y_idx"],
            data["x_idx"],
        )
        _ = model.inference_with_anchor(
            X,
            y_anchor=y_anchor,
            x_anchor=x_anchor,
            padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
        )

    model.use_kernelized_instance_norm_for_whole_model(
        padding=config["INFERENCE_SETTING"]["KIN_PADDING"]
    )

    for idx, data in enumerate(dataloader):
        X, X_path, y_anchor, x_anchor = (
            data["X_img"],
            data["X_path"],
            data["y_idx"],
            data["x_idx"],
        )
        Y_fake = model.inference_with_anchor(
            X,
            y_anchor=y_anchor,
            x_anchor=x_anchor,
            padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
        )
        test_output_tensor = reverse_image_normalize(Y_fake)

        save_path_root = os.path.join(
            config["EXPERIMENT_ROOT_PATH"],
            config["EXPERIMENT_NAME"],
            "test"
        )
        save_dir = os.path.join(
            save_path_root,
            config["INFERENCE_SETTING"]["NORMALIZATION"],
            config["INFERENCE_SETTING"]["MODEL_VERSION"],
            f"{config['INFERENCE_SETTING']['KIN_KERNEL']}_"
            f"{config['INFERENCE_SETTING']['KIN_PADDING']}_pytest",
        )
        
        os.makedirs(save_dir, exist_ok=True)

        test_output_path = os.path.join(
            save_dir,
            f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
        )
        save_image(
            test_output_tensor,
            test_output_path
        )
        test_output = np.array(
            Image.open(test_output_path).convert("RGB")
        )
        # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # test_output = make_grid(test_output_tensor).mul(255).add_(0.5).clamp_(
        #     0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

        expected_output_path = os.path.join(
            save_path_base,
            f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
        )
        expected_output = np.array(
            Image.open(expected_output_path).convert("RGB")
        )
        print(test_output == pytest.approx(expected_output, abs=1e-6))
        assert test_output == pytest.approx(expected_output, abs=1e-6)
