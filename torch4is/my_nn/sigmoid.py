import torch
import torch.nn as nn


class MySigmoid(nn.Module):
    """nn.Sigmoid"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
