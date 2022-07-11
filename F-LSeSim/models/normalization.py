import torch.nn as nn

from models.kin import KernelizedInstanceNorm
from models.tin import ThumbInstanceNorm


def get_normalization_layer(num_features, normalization="kin"):
    if normalization == "kin":
        return KernelizedInstanceNorm(num_features=num_features)
    elif normalization == "tin":
        return ThumbInstanceNorm(num_features=num_features)
    elif normalization == "in":
        return nn.InstanceNorm2d(num_features)
    else:
        raise NotImplementedError
