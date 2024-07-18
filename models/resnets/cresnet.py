import torch
import torch.nn as nn
import torch.nn.functional as F

from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d, CliffordConv3d


class CliffordG3GroupNorm(nn.Module):
    """
    A module that applies group normalization to vectors in G3.

    Args:
        num_groups (int): Number of groups to normalize over.
        num_features (int): Number of features in the input.
        num_blades (int): Number of blades in the input.
        scale_norm (bool, optional): If True, the output is scaled by the norm of the input. Defaults to False.
    """

    def __init__(self, num_groups, num_features, num_blades, scale_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features, num_blades))
        self.num_groups = num_groups
        self.scale_norm = scale_norm
        self.num_blades = num_blades
        self.num_features = num_features

    def forward(self, x):
        N, C, *D, I = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1, I)
        mean = x.mean(-2, keepdim=True)
        x = x - mean
        if self.scale_norm:
            norm = x.norm(dim=-1, keepdim=True).mean(dim=-2, keepdims=True)
            x = x / norm

        x = x.view(len(x), self.num_features, -1, self.num_blades)

        return (x * self.weight[None, :, None, None] + self.bias[None, :, None]).view(
            N, C, *D, I
        )


class CliffordBasicBlock(nn.Module):
    def __init__(self, metric, in_planes, planes, kernel_size, norm=True):
        super().__init__()
        conv_operator = CliffordConv2d if len(metric) == 2 else CliffordConv3d
        self.activation = F.gelu

        self.conv1 = conv_operator(
            metric, in_planes, planes, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = conv_operator(
            metric, planes, planes, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # Use GroupNorm with number of groups = 1, which is similar to LayerNorm for convolutional layers
        self.groupnorm = (
            CliffordG3GroupNorm(1, planes, 2 ** len(metric)) if norm else nn.Identity()
        )

        if in_planes != planes:
            self.shortcut = nn.Sequential(
                conv_operator(metric, in_planes, planes, kernel_size=1),
                (
                    CliffordG3GroupNorm(1, planes, 2 ** len(metric))
                    if norm
                    else nn.Identity()
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.groupnorm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.groupnorm(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class CliffordResNet(nn.Module):

    def __init__(
        self,
        metric: tuple,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        kernel_size: int = 7,
        norm: bool = True,
        blocks: tuple = (2, 2, 2, 2),
        make_channels: bool = False,
    ):
        super().__init__()
        in_channels = 1 if make_channels else time_history
        out_channels = 1 if make_channels else time_future

        conv_operator = CliffordConv2d if len(metric) == 2 else CliffordConv3d
        self.activation = F.gelu

        self.conv_in1 = conv_operator(
            metric,
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = conv_operator(
            metric,
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = conv_operator(
            metric,
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = conv_operator(
            metric,
            hidden_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList([])
        for num_blocks in blocks:
            for _ in range(num_blocks):
                self.layers.append(
                    CliffordBasicBlock(
                        metric,
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        norm=norm,
                    )
                )

    def __repr__(self):
        return "Clifford ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, *space, channels)
        x = self.conv_in1(x)
        x = self.activation(x)
        x = self.conv_in2(x)
        x = self.activation(x)

        # Apply residual layers.
        for layer in self.layers:
            x = layer(x)

        x = self.conv_out1(x)
        x = self.activation(x)
        x = self.conv_out2(x)

        return x
