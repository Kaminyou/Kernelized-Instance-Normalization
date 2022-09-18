import os

import numpy as np
import pytest
import torch
from models.model import get_model
from models.kin import KernelizedInstanceNorm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.dataset import XInferenceDataset
from utils.util import (read_yaml_config, reverse_image_normalize,
                        test_transforms)


@pytest.fixture()
def config():
    config_path = os.path.join(
        "./test_data",
        "configs",
        "config_lung_lesion_for_test_cyclegan.yaml"
    )
    config = read_yaml_config(config_path)

    return config


@pytest.fixture()
def in_model(config):
    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        norm_cfg={'type': 'in'},
        isTrain=False,
    )
    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    return model


@pytest.fixture()
def kin_model(config):
    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        norm_cfg=config['INFERENCE_SETTING']['NORMALIZATION'],
        isTrain=False,
    )
    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])
    model.eval()

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
def in_expected_outputs(config):
    expected_output_dir = os.path.join(
        config["INFERENCE_SETTING"]["TEST_DIR_Y"],
        config["EXPERIMENT_NAME"],
        "in",
        f"{config['INFERENCE_SETTING']['MODEL_VERSION']}",

    )
    expected_output_files = sorted(
        os.listdir(expected_output_dir)
    )

    expected_outputs = []
    for expected_output_file in expected_output_files:
        expected_outputs.append(np.array(
            Image.open(os.path.join(
                expected_output_dir,
                expected_output_file
            )).convert("RGB")
        ))

    return expected_outputs


@pytest.fixture()
def kin_expected_outputs(config):
    expected_output_dir = os.path.join(
        config["INFERENCE_SETTING"]["TEST_DIR_Y"],
        config["EXPERIMENT_NAME"],
        "kin",
        config["INFERENCE_SETTING"]["MODEL_VERSION"],
        f"{config['INFERENCE_SETTING']['NORMALIZATION']['KERNEL_TYPE']}_"
        f"{config['INFERENCE_SETTING']['NORMALIZATION']['PADDING']}",

    )
    expected_output_files = sorted(
        os.listdir(expected_output_dir)
    )

    expected_outputs = []
    for expected_output_file in expected_output_files:
        expected_outputs.append(np.array(
            Image.open(os.path.join(
                expected_output_dir,
                expected_output_file
            )).convert("RGB")
        ))

    return expected_outputs


def test_inference(in_model, dataloader, in_expected_outputs):
    """
        Integration testing for IN inferece
    """

    for idx, data in enumerate(dataloader):
        X, _ = data["X_img"], data["X_path"]
        Y_fake = in_model.inference(X)
        test_output_tensor = reverse_image_normalize(Y_fake)

        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        test_output = make_grid(test_output_tensor).mul(255).add_(0.5).clamp_(
            0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        expected_output = in_expected_outputs[idx]

        assert test_output == pytest.approx(expected_output)


def test_inference_with_anchor(
    config,
    kin_model,
    dataset,
    dataloader,
    kin_expected_outputs
):
    """
        Integration testing for KIN inferece
    """

    y_anchor_num, x_anchor_num = dataset.get_boundary()

    kin_model.init_kernelized_instance_norm_for_whole_model(
        y_anchor_num=y_anchor_num + 1,
        x_anchor_num=x_anchor_num + 1,
    )

    for idx, data in enumerate(dataloader):
        X, _, y_anchor, x_anchor = (
            data["X_img"],
            data["X_path"],
            data["y_idx"],
            data["x_idx"],
        )
        _ = kin_model.inference_with_anchor(
            X,
            y_anchor=y_anchor,
            x_anchor=x_anchor,
            mode=KernelizedInstanceNorm.Mode.PHASE_CACHING,
        )

    for idx, data in enumerate(dataloader):
        X, _, y_anchor, x_anchor = (
            data["X_img"],
            data["X_path"],
            data["y_idx"],
            data["x_idx"],
        )
        Y_fake = kin_model.inference_with_anchor(
            X,
            y_anchor=y_anchor,
            x_anchor=x_anchor,
            mode=KernelizedInstanceNorm.Mode.PHASE_INFERENCE,
        )
        test_output_tensor = reverse_image_normalize(Y_fake)

        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        test_output = make_grid(test_output_tensor).mul(255).add_(0.5).clamp_(
            0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        expected_output = kin_expected_outputs[idx]

        assert test_output == pytest.approx(expected_output)
