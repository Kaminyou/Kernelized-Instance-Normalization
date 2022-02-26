import itertools

import numpy as np
import torch
import torch.nn as nn


class KernelizedInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True, device="cuda"):
        super(KernelizedInstanceNorm, self).__init__()
        self.normal_instance_normalization = False # if use normal instance normalization during evaluation mode
        self.collection_mode = False # if collecting instance normalization mean and std during evaluation mode'
        self.out_channels = out_channels
        self.device = device
        if affine == True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)).to(device)
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)).to(device)

    def init_collection(self, y_anchor_num, x_anchor_num):
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels).to(self.device)
        self.std_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels).to(self.device)

    def init_kernel(self, kernel=(torch.ones(1,1,3,3)/9)):
        kernel = kernel.to(self.device)
        self.kernel = kernel

    def collection(self, instance_means, instnace_stds, y_anchors, x_anchors):
        instance_means = instance_means.squeeze(-1).squeeze(-1)
        instnace_stds = instnace_stds.squeeze(-1).squeeze(-1)
        for instance_mean, instnace_std, y_anchor, x_anchor in zip(instance_means, instnace_stds, y_anchors, x_anchors):
            self.mean_table[y_anchor, x_anchor, :] = instance_mean
            self.std_table[y_anchor, x_anchor, :] = instnace_std

    def query_neighbors(self, y_anchor, x_anchor, padding=1):
        """
        return_anchors:: [top, down], [left, right] all are inclusive
        """
        y_anchor_top = max(0, y_anchor - padding)
        y_anchor_down = min(self.y_anchor_num, y_anchor + padding)
        x_anchor_left = max(0, x_anchor - padding)
        x_anchor_right = min(self.x_anchor_num, x_anchor + padding)
        return [y_anchor_top, y_anchor_down, x_anchor_left, x_anchor_right]

    def pad_table(self, padding):
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(self.mean_table.permute(2, 0, 1).unsqueeze(0)) # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(self.std_table.permute(2, 0, 1).unsqueeze(0)) # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def __multiply_kernel(self, x_stat):
        # self.kernel = [1,1,H,W] || x_stat = [1,C,H,W]
        assert self.kernel.shape[2:] == x_stat.shape[2:]
        x_stat = x_stat * self.kernel # [1,C,H,W] = [1,C,H,W] * [1,1,H,W]
        x_stat = x_stat.flatten(start_dim=2).sum(dim=2) # [1, C, H, W] -> [1, C, H * W] -> [1, C]
        x_stat = x_stat.unsqueeze(-1).unsqueeze(-1) # [1, C] -> [1, C, 1, 1]
        return x_stat

    def forward_normal(self, x):
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std #* self.weight + self.bias
        return x

    def forward(self, x, y_anchor=None, x_anchor=None, padding=1):
        if self.training or self.normal_instance_normalization:
            return self.forward_normal(x)

        else:
            assert y_anchor != None
            assert x_anchor != None

            if self.collection_mode:
                x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
                self.collection(instance_means=x_mean, instnace_stds=x_std, y_anchors=y_anchor, x_anchors=x_anchor)

            else:
                assert x.shape[0] == 1 # currently, could support batch size = 1 for kernelized instance normalization
                top = y_anchor
                down = y_anchor + 2 * padding + 1
                left = x_anchor
                right = x_anchor + 2 *padding + 1
                x_mean = self.padded_mean_table[:,:,top:down, left:right] # 1, C, H, W
                x_std = self.padded_std_table[:,:,top:down, left:right] # 1, C, H, W
                x_mean = self.__multiply_kernel(x_mean)
                x_std = self.__multiply_kernel(x_std)

            x = (x - x_mean) / x_std * self.weight + self.bias
            return x

def not_use_kernelized_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True

def init_kernelized_instance_norm(model, y_anchor_num, x_anchor_num, kernel=(torch.ones(1,1,3,3)/9)):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)
            layer.init_kernel(kernel=kernel)

def use_kernelized_instance_norm(model, padding=1):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


"""
USAGE
    support a dataset with a dataloader would return (x, y_anchor, x_anchor) each time

    kin = KernelizedInstanceNorm()

    [TRAIN] anchors are not used during training
    kin.train()
    for (x, _, _) in dataloader:
        kin(x)

    [COLLECT] anchors are required and any batch size is allowed
    kin.eval()
    init_kernelized_instance_norm(kin, y_anchor_num=$y_anchor_num, x_anchor_num=$x_anchor_num, kernel=torch.ones(3,3))
    for (x, y_anchor, x_anchor) in dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor)

    [INFERENCE] anchors are required and batch size is limited to 1 !!
    kin.eval()
    use_kernelized_instance_norm(kin)
    for (x, y_anchor, x_anchor) in dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor, padding=$padding)

    [INFERENCE WITH NORMAL INSTANCE NORMALIZATION] anchors are not required
    kin.eval()
    not_use_kernelized_instance_norm(kin)
    for (x, _, _) in dataloader:
        kin(x)
"""

if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset

    class TestDataset(Dataset):
        def __init__(self, y_anchor_num=10, x_anchor_num=10):
            self.y_anchor_num = y_anchor_num
            self.x_anchor_num = x_anchor_num
            self.anchors = list(itertools.product(np.arange(0, y_anchor_num), np.arange(0, x_anchor_num)))

        def __len__(self):
            return len(self.anchors)

        def __getitem__(self, idx):
            x = torch.randn(3, 512, 512)
            y_anchor, x_anchor = self.anchors[idx]
            return (x, y_anchor, x_anchor)


    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=5)

    kin = KernelizedInstanceNorm(out_channels=3)
    kin.eval()
    init_kernelized_instance_norm(kin, y_anchor_num=10, x_anchor_num=10, kernel=torch.ones(3,3))

    for (x, y_anchor, x_anchor) in test_dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor)

    use_kernelized_instance_norm(kin)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    for (x, y_anchor, x_anchor) in test_dataloader:
        x = kin(x, y_anchor=y_anchor, x_anchor=x_anchor)
        print(x.shape)
