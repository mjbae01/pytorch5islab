from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv2d(nn.Module):
    """nn.Conv2d"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 groups: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # for simplicity, we support square-size kernel/stride/padding and zero padding only.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if (out_channels % groups != 0) or (in_channels % groups != 0):
            raise ValueError(f"[ERROR:NN] Groups is not divisible.")
        self.groups = groups

        self.use_bias = bias
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=2.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def calculate_output_shape(self, h: int, w: int) -> Tuple[int, int]:
        """Calculate output shape with given 2D input (h, w)"""
        h_out = (h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        w_out = (w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        return h_out, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, in_channels, h, w)
        :return:        (batch_size, out_channels, new_h, new_w)
        """
        y = F.conv2d(x,
                     self.weight,
                     self.bias,
                     self.stride,
                     self.padding,
                     self.dilation,
                     self.groups)
        return y
