import torch.nn as nn

from models.kin import KernelizedInstanceNorm
from models.tin import ThumbInstanceNorm


def get_normalization_layer(out_channels, normalization="kin"):
    if normalization == "kin":
        return ThumbInstanceNorm(out_channels=out_channels)
    elif normalization == "tin":
        return KernelizedInstanceNorm(out_channels=out_channels)
    elif normalization == "in":
        return nn.InstanceNorm2d(out_channels)
    else:
        raise NotImplementedError
