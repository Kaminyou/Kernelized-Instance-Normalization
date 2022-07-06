import numpy as np
import torch
from scipy import signal


def gkern(kernlen=1, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def get_kernel(padding=1, gaussian_std=3, mode="constant"):
    kernel_size = padding * 2 + 1
    if mode == "constant":
        kernel = torch.ones(kernel_size, kernel_size)
        kernel = kernel / (kernel_size * kernel_size)

    elif mode == "gaussian":
        kernel = gkern(kernel_size, std=gaussian_std)
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel.astype(np.float32))

    else:
        raise NotImplementedError

    return kernel