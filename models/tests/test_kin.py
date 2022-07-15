import numpy as np
import pytest
import torch

from ..kin import KernelizedInstanceNorm


def normalize(x):
    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
    return (x - mean) / std


def test_forward_normal():
    layer = KernelizedInstanceNorm(num_features=3, device='cpu')
    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    expected = normalize(x)

    check = layer.forward_normal(torch.FloatTensor(x))

    assert check.numpy() == pytest.approx(expected, abs=1e-6)


def test_init_collection():
    layer = KernelizedInstanceNorm(num_features=3, device='cpu')
    layer.init_collection(y_anchor_num=10, x_anchor_num=9)

    expected_mean_table = np.zeros(shape=(10, 9, 3))
    expected_std_table = np.zeros(shape=(10, 9, 3))

    np.testing.assert_array_equal(layer.mean_table.numpy(), expected_mean_table)
    np.testing.assert_array_equal(layer.std_table.numpy(), expected_std_table)


def test_forward_with_normal_instance_normalization():
    layer = KernelizedInstanceNorm(num_features=3, device='cpu')
    layer.normal_instance_normalization = True
    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    expected = normalize(x)

    check = layer.forward_normal(torch.FloatTensor(x))

    assert check.numpy() == pytest.approx(expected, abs=1e-6)


def test_forward_with_collection_mode():
    layer = KernelizedInstanceNorm(num_features=3, kernel_type='constant', device='cpu').eval()
    layer.collection_mode = True
    layer.normal_instance_normalization = False

    layer.init_collection(y_anchor_num=3, x_anchor_num=3)

    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    std, mean = torch.std_mean(x, dim=(2, 3))

    expected_mean_table = np.zeros(shape=(3, 3, 3), dtype=np.float32)
    expected_std_table = np.zeros(shape=(3, 3, 3), dtype=np.float32)

    expected_mean_table[0, 0] = mean
    expected_std_table[0, 0] = std

    check = layer.forward(x, x_anchor=0, y_anchor=0)

    assert check.detach().numpy() == pytest.approx(normalize(x).numpy(), abs=1e-6)
    assert layer.mean_table.numpy() == pytest.approx(expected_mean_table)
    assert layer.std_table.numpy() == pytest.approx(expected_std_table)


def test_forward_with_kernelized():
    layer = KernelizedInstanceNorm(num_features=3, kernel_type='constant', device='cpu').eval()
    layer.collection_mode = True
    layer.normal_instance_normalization = False

    layer.init_collection(y_anchor_num=3, x_anchor_num=3)

    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    layer.forward(x, x_anchor=1, y_anchor=1)

    layer.collection_mode = False

    check = layer.forward(x, x_anchor=1, y_anchor=1)
    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)

    mean /= 9
    std /= 9

    expected = (x - mean) / std

    assert check.detach().numpy() == pytest.approx(expected)
