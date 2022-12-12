import torch
import torch.nn as nn
import torch.nn.functional as F


class LBlock(torch.nn.Module):
    """
    L Block: A modified residual block designed specifically for increasing the
             number of channels of its respective output. Used in Latent Stack.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.

    Returns:
        torch.tensor
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(LBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(
            self.in_channels, self.out_channels - self.in_channels, 1
        )
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

    def forward(self, inputs) -> torch.Tensor:
        x0 = self.relu(inputs)
        x0 = self.conv3_1(x0)
        x0 = self.relu(x0)
        x0 = self.conv3_2(x0)

        if self.in_channels < self.out_channels:
            x1 = self.conv1_1(inputs)
            x1 = torch.cat([inputs, x1], dim=1)
        else:
            x1 = inputs

        return x0 + x1


class Attention(torch.nn.Module):
    """
    Attention module
    """

    def __init__(
        self, in_channels: int, out_channels: int, ratio_kq: int = 8, ratio_v: int = 8
    ):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v

        # Compute K,Q and V using 1x1 convolution
        self.query = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        self.key = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        self.value = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // self.ratio_v,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        self.final = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels // 8
        )

        # Trainable parameter:
        self.gamma = nn.Parameter(torch.zeros([1]))

    def einsum(self, q, k, v):
        """
        Apply the attention operator to tensors of shape [b, c, h, w]...
        """
        # Reshape 3D tensors to 2D tensors with L = h x w, thus: [b, c, h, w] -> [b, c, L]
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # Einsum should be applied per batch.
        out = self.einsum(q, k, v)
        out = self.gamma * self.final(out)
        # Return with residual.
        return out + inputs
