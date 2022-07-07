import numpy as np
import pytest
import torch

from ..kin import KernelizedInstanceNorm


def normalize(x):
    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
    return (x - mean) / std


def test_forward_normal():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu')
    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    expected = normalize(x)

    check = layer.forward_normal(torch.FloatTensor(x))

    assert check.numpy() == pytest.approx(expected, abs=1e-6)


def test_init_kernel():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu')
    layer.init_kernel(kernel_padding=1, kernel_mode='constant')

    expected = np.ones(shape=(3, 3), dtype=np.float32) / 9

    assert layer.kernel.numpy() == pytest.approx(expected)


def test_init_collection():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu')
    layer.init_collection(y_anchor_num=10, x_anchor_num=9)

    expected_mean_table = np.zeros(shape=(10, 9, 3))
    expected_std_table = np.zeros(shape=(10, 9, 3))

    np.testing.assert_array_equal(layer.mean_table.numpy(), expected_mean_table)
    np.testing.assert_array_equal(layer.std_table.numpy(), expected_std_table)


def test_pad_table():
    layer = KernelizedInstanceNorm(out_channels=1, device='cpu')

    table = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=np.float32
    ).reshape(2, 2, 1)

    expected_table = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.float32
    ).reshape(4, 4, 1)

    layer.mean_table = torch.FloatTensor(table)
    layer.std_table = torch.FloatTensor(table)

    layer.pad_table(padding=1)

    expected_padded_mean_table = expected_table.transpose(2, 0, 1).reshape(1, 1, 4, 4)
    expected_padded_std_table = expected_table.transpose(2, 0, 1).reshape(1, 1, 4, 4)

    np.testing.assert_array_equal(layer.padded_mean_table.numpy(), expected_padded_mean_table)
    np.testing.assert_array_equal(layer.padded_std_table.numpy(), expected_padded_std_table)


def test_forward_with_normal_instance_normalization():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu')
    layer.normal_instance_normalization = True
    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    expected = normalize(x)

    check = layer.forward_normal(torch.FloatTensor(x))

    assert check.numpy() == pytest.approx(expected, abs=1e-6)


def test_forward_with_collection_mode():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu').eval()
    layer.collection_mode = True
    layer.normal_instance_normalization = False

    layer.init_collection(y_anchor_num=3, x_anchor_num=3)
    layer.init_kernel(kernel_padding=1, kernel_mode='constant')

    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    std, mean = torch.std_mean(x, dim=(2, 3))

    expected_mean_table = np.zeros(shape=(3, 3, 3), dtype=np.float32)
    expected_std_table = np.zeros(shape=(3, 3, 3), dtype=np.float32)

    expected_mean_table[0, 0] = mean
    expected_std_table[0, 0] = std

    check = layer.forward(x, x_anchor=0, y_anchor=0, padding=1)

    assert check.detach().numpy() == pytest.approx(normalize(x).numpy(), abs=1e-6)
    assert layer.mean_table.numpy() == pytest.approx(expected_mean_table)
    assert layer.std_table.numpy() == pytest.approx(expected_std_table)


def test_forward_with_kernelized():
    layer = KernelizedInstanceNorm(out_channels=3, device='cpu').eval()
    layer.collection_mode = True
    layer.normal_instance_normalization = False

    layer.init_collection(y_anchor_num=3, x_anchor_num=3)
    layer.init_kernel(kernel_padding=1, kernel_mode='constant')

    x = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
    x = torch.FloatTensor(x)

    layer.forward(x, x_anchor=1, y_anchor=1, padding=1)

    layer.collection_mode = False
    layer.pad_table(1)

    check = layer.forward(x, x_anchor=1, y_anchor=1, padding=1)
    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)

    mean /= 9
    std /= 9

    expected = (x - mean) / std

    assert check.detach().numpy() == pytest.approx(expected)
