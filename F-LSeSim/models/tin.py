import torch
import torch.nn as nn


class ThumbInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True):
        super(ThumbInstanceNorm, self).__init__()
        self.thumb_mean = None
        self.thumb_std = None
        self.normal_instance_normalization = False
        self.collection_mode = False
        if affine == True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, x):
        if self.training or self.normal_instance_normalization:
            x_mean, x_std = self.calc_mean_std(x)
            x = (x - x_mean) / x_std * self.weight + self.bias
            return x
        else:
            if self.collection_mode:
                assert x.shape[0] == 1
                x_mean, x_std = self.calc_mean_std(x)
                self.thumb_mean = x_mean
                self.thumb_std = x_std

            x = (x - self.thumb_mean) / self.thumb_std * self.weight + self.bias
            return x

def not_use_thumbnail_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, ThumbInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True

def init_thumbnail_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, ThumbInstanceNorm):
            layer.collection_mode = True

def use_thumbnail_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, ThumbInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = False
