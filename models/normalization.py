from copy import deepcopy
from typing import Any, Dict

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


def make_norm_layer(norm_cfg: Dict[str, Any], **kwargs: Any):
    """

    Args:
        norm_cfg (Dict[str, Any]): A dict of keyword arguments of normalization layer.
            It must have a key 'type' to specify which normalization layers will be used.
        **kwargs (Any): The keyword arguments are used to overwrite `norm_cfg`.

    Returns:
        nn.Module: A layer object.
    """
    norm_cfg = deepcopy(norm_cfg)
    norm_cfg.update(kwargs)

    if 'type' not in norm_cfg:
        raise ValueError('"type" wasn\'t specified.')

    norm_type = norm_cfg['type']
    del norm_cfg['type']

    if norm_type == 'in':
        return nn.InstanceNorm2d(**norm_cfg)
    elif norm_type == 'tin':
        return ThumbInstanceNorm(**norm_cfg)
    elif norm_type == 'kin':
        return KernelizedInstanceNorm(**norm_cfg)
    else:
        raise ValueError(f'Unknown norm type: {norm_type}.')
