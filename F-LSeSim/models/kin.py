import numpy as np
import torch
import torch.nn as nn
from .util import get_kernel


class KernelizedInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True, device="cuda"):
        super(KernelizedInstanceNorm, self).__init__()

        # if use normal instance normalization during evaluation mode
        self.normal_instance_normalization = False

        # if collecting instance normalization mean and std
        # during evaluation mode'
        self.collection_mode = False

        self.out_channels = out_channels
        self.device = device
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)

    def init_collection(self, y_anchor_num, x_anchor_num):
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )

    def init_kernel(self, kernel_padding, kernel_mode):
        kernel = get_kernel(padding=kernel_padding, mode=kernel_mode)
        self.kernel = kernel.to(self.device)

    def pad_table(self, padding):
        # modify
        # padded table shape inconsisency
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x):
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(self, x, y_anchor=None, x_anchor=None, padding=1):
        if self.training or self.normal_instance_normalization:
            return self.forward_normal(x)

        else:
            assert y_anchor is not None
            assert x_anchor is not None

            if self.collection_mode:
                x_std, x_mean = torch.std_mean(x, dim=(2, 3))  # [B, C]
                # x_anchor, y_anchor = [B], [B]
                # table = [H, W, C]
                # update std and mean to corresponing coordinates
                self.mean_table[y_anchor, x_anchor] = x_mean
                self.std_table[y_anchor, x_anchor] = x_std
                x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
                x_std = x_std.unsqueeze(-1).unsqueeze(-1)

            else:

                def multiply_kernel(x):
                    x = x * self.kernel  # [1, C, H, W] * [H, W] = [1, C, H, W]
                    x = x.sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                    return x

                # currently, could support batch size = 1 for
                # kernelized instance normalization
                assert x.shape[0] == 1

                top = y_anchor
                down = y_anchor + 2 * padding + 1
                left = x_anchor
                right = x_anchor + 2 * padding + 1
                x_mean = self.padded_mean_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W
                x_std = self.padded_std_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W
                x_mean = multiply_kernel(x_mean)
                x_std = multiply_kernel(x_std)

            x = (x - x_mean) / x_std * self.weight + self.bias
            return x


def not_use_kernelized_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True


def init_kernelized_instance_norm(
    model, y_anchor_num, x_anchor_num, kernel_padding, kernel_mode
):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )
            layer.init_kernel(
                kernel_padding=kernel_padding, kernel_mode=kernel_mode
            )


def use_kernelized_instance_norm(model, padding=1):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


"""
USAGE
    support a dataset with a dataloader would return
    (x, y_anchor, x_anchor) each time
    kin = KernelizedInstanceNorm()
    [TRAIN] anchors are not used during training
    kin.train()
    for (x, _, _) in dataloader:
        kin(x)
    [COLLECT] anchors are required and any batch size is allowed
    kin.eval()
    init_kernelized_instance_norm(
        kin, y_anchor_num=$y_anchor_num,
        x_anchor_num=$x_anchor_num,
        kernel_padding=$kernel_padding,
        kernel_mode=$kernel_mode,
    )
    for (x, y_anchor, x_anchor) in dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor)
    [INFERENCE] anchors are required and batch size is limited to 1 !!
    kin.eval()
    use_kernelized_instance_norm(kin, kernel_padding=$kernel_padding)
    for (x, y_anchor, x_anchor) in dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor, padding=$padding)
    [INFERENCE WITH NORMAL INSTANCE NORMALIZATION] anchors are not required
    kin.eval()
    not_use_kernelized_instance_norm(kin)
    for (x, _, _) in dataloader:
        kin(x)
"""

if __name__ == "__main__":
    import itertools

    from torch.utils.data import DataLoader, Dataset

    class TestDataset(Dataset):
        def __init__(self, y_anchor_num=10, x_anchor_num=10):
            self.y_anchor_num = y_anchor_num
            self.x_anchor_num = x_anchor_num
            self.anchors = list(
                itertools.product(
                    np.arange(0, y_anchor_num), np.arange(0, x_anchor_num)
                )
            )

        def __len__(self):
            return len(self.anchors)

        def __getitem__(self, idx):
            x = torch.randn(3, 512, 512)
            y_anchor, x_anchor = self.anchors[idx]
            return (x, y_anchor, x_anchor)

    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=5)

    kin = KernelizedInstanceNorm(out_channels=3, device="cpu")
    kin.eval()
    init_kernelized_instance_norm(
        kin,
        y_anchor_num=10,
        x_anchor_num=10,
        kernel_padding=1,
        kernel_mode="constant",
    )

    for (x, y_anchor, x_anchor) in test_dataloader:
        kin(x, y_anchor=y_anchor, x_anchor=x_anchor)

    use_kernelized_instance_norm(kin, kernel_padding=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    for (x, y_anchor, x_anchor) in test_dataloader:
        x = kin(x, y_anchor=y_anchor, x_anchor=x_anchor, padding=1)
        print(x.shape)