import torch
import torch.nn as nn

from utils.util import get_kernel


class KernelizedInstanceNorm(nn.Module):
    """

    Attributes:
        num_features (int): The number of input features.
        kernel_size (int): The size of kernel used to kernelize.
        kernel_type (str): The kernel type. It must be either 'constant' or 'gaussian'.
        padding (int): The padding size of cache table. It must be `(kernel_size - 1) / 2`.
        eps (float): A value added to the denominator for numerical stability.
        affine (bool): A boolean value that when set to True, this module has learnable
            affine parameters, initialized the same way as done for batch normalization.
        device (str): The device to place the variables.
    """

    def __init__(
        self,
        num_features,
        kernel_size=3,
        kernel_type='gaussian',
        padding=1,
        eps=0,
        affine=True,
        device="cuda",
    ):
        super(KernelizedInstanceNorm, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be an odd number. But got {kernel_size}.',
            )

        if padding != (kernel_size - 1) // 2:
            raise ValueError(
                f'padding must be (kernel_size - 1) / 2. But got {padding}.'
            )

        # if use normal instance normalization during evaluation mode
        self.normal_instance_normalization = False

        # if collecting instance normalization mean and std
        # during evaluation mode'
        self.collection_mode = False

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.padding = padding
        self.eps = eps
        self.affine = affine
        self.device = device

        self.kernel = get_kernel(
            padding=(self.kernel_size - 1) // 2,
            mode=self.kernel_type,
        ).to(self.device)

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, num_features, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, num_features, 1, 1), requires_grad=True)
            ).to(device)

    def init_collection(self, y_anchor_num, x_anchor_num):
        # TODO: y_anchor_num => grid_height, x_anchor_num => grid_width
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.num_features
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.num_features
        ).to(
            self.device
        )

    def pad_table(self):
        # modify
        # padded table shape inconsisency
        # TODO: Don't permute the dimensions
        pad_func = nn.ReplicationPad2d((self.padding, self.padding, self.padding, self.padding))
        self.padded_mean_table = pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x):
        x_var, x_mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.sqrt(x_var + self.eps)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(self, x, y_anchor=None, x_anchor=None):
        # TODO: Do not reply on self.training
        if self.training or self.normal_instance_normalization:
            return self.forward_normal(x)

        else:
            assert y_anchor is not None
            assert x_anchor is not None

            if self.collection_mode:
                x_var, x_mean = torch.var_mean(x, dim=(2, 3))  # [B, C]
                x_std = torch.sqrt(x_var + self.eps)
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
                down = y_anchor + 2 * self.padding + 1
                left = x_anchor
                right = x_anchor + 2 * self.padding + 1
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


def init_kernelized_instance_norm(model, y_anchor_num, x_anchor_num):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )


def use_kernelized_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, KernelizedInstanceNorm):
            layer.pad_table()
            layer.collection_mode = False
            layer.normal_instance_normalization = False
