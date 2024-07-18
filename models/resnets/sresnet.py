import math

import torch
from torch import nn

import escnn
from escnn import gspaces, group

import os, sys

sys.path.append(os.path.join(os.getcwd(), "../../.."))

from algebra.torched.cliffordalgebra import CliffordAlgebra


def gt_to_mv(x):
    n_channels = len(x.type)
    n_components = 4
    x = x.tensor
    x = x.reshape(x.shape[0], n_channels, n_components, *x.shape[2:])
    x = x.permute(0, 1, 3, 4, 2)
    return x


def mv_to_gt(x, type):
    x = x.permute(0, 1, 4, 2, 3)
    x = x.reshape(x.shape[0], -1, *x.shape[3:])
    return type(x)


def get_type(planes):
    gspace = gspaces.flipRot2dOnR2(-1)
    scalar, pseudoscalar, vector = gspace.fibergroup.irreps()[:3]
    representation = group.directsum([scalar, vector, pseudoscalar])
    return gspace.type(*planes * [representation])


class MVGELU(nn.Module):
    def forward(self, input):
        gates = input[..., [0]]
        weights = torch.sigmoid(
            math.sqrt(2 / torch.pi) * (2 * (gates + 0.044715 * torch.pow(gates, 3)))
        )
        return weights * input


class MVLayerNorm(nn.Module):
    def __init__(self, algebra):
        super(MVLayerNorm, self).__init__()
        self.algebra = algebra

    def forward(self, input):
        norms = self.algebra.norm(input).mean(dim=1, keepdim=True)
        outputs = input / norms
        return outputs


class SteerableBasicBlock(nn.Module):
    def __init__(self, metric, in_planes, planes, kernel_size, norm=True):
        super().__init__()
        self.metric = metric
        conv_operator = escnn.nn.R2Conv if len(metric) == 2 else escnn.nn.R3Conv
        self.in_type = get_type(in_planes)
        self.hidden_type = get_type(planes)

        self.activation = MVGELU()
        self.norm = MVLayerNorm(CliffordAlgebra(metric))

        self.conv1 = conv_operator(
            self.in_type,
            self.hidden_type,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = conv_operator(
            self.hidden_type,
            self.hidden_type,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        if in_planes != planes:
            self.shortcut_conv = conv_operator(
                self.in_type, self.hidden_type, kernel_size=1
            )
            self.shortcut_norm = (
                MVLayerNorm(CliffordAlgebra(metric)) if norm else nn.Identity()
            )
        else:
            self.shortcut_conv = nn.Identity()
            self.shortcut_norm = nn.Identity()

        self.initialize()

    def initialize(self):
        escnn.nn.init.generalized_he_init(
            self.conv1.weights.data, self.conv1.basisexpansion
        )
        escnn.nn.init.generalized_he_init(
            self.conv2.weights.data, self.conv2.basisexpansion
        )
        if not isinstance(self.shortcut_conv, nn.Identity):
            escnn.nn.init.generalized_he_init(
                self.shortcut_conv.weights.data, self.shortcut_conv.basisexpansion
            )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        x = mv_to_gt(x, self.in_type)
        out = self.conv1(x)
        out = gt_to_mv(out)
        out = self.norm(out)
        out = self.activation(out)
        out = mv_to_gt(out, self.hidden_type)
        out = self.conv2(out)
        out = gt_to_mv(out)
        out = self.norm(out)
        shortcut = self.shortcut_conv(x)
        shortcut = gt_to_mv(shortcut)
        shortcut = self.shortcut_norm(shortcut)
        out += shortcut
        out = self.activation(out)
        return out


class SteerableResNet(nn.Module):

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
        self.in_type = get_type(time_history)
        self.hidden_type = get_type(hidden_channels)
        self.out_type = get_type(time_future)

        conv_operator = escnn.nn.R2Conv if len(metric) == 2 else escnn.nn.R3Conv
        self.activation = MVGELU()

        self.conv_in1 = conv_operator(self.in_type, self.hidden_type, kernel_size=1)
        self.conv_in2 = conv_operator(self.hidden_type, self.hidden_type, kernel_size=1)

        self.conv_out1 = conv_operator(
            self.hidden_type, self.hidden_type, kernel_size=1
        )
        self.conv_out2 = conv_operator(self.hidden_type, self.out_type, kernel_size=1)

        self.layers = nn.ModuleList([])
        for num_blocks in blocks:
            for _ in range(num_blocks):
                self.layers.append(
                    SteerableBasicBlock(
                        metric,
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        norm=norm,
                    )
                )

    def __repr__(self):
        return "Steerable ResNet"

    def initialize(self):
        escnn.nn.init.generalized_he_init(
            self.conv_in1.weights.data, self.conv_in1.basisexpansion
        )
        escnn.nn.init.generalized_he_init(
            self.conv_in2.weights.data, self.conv_in2.basisexpansion
        )
        escnn.nn.init.generalized_he_init(
            self.conv_out1.weights.data, self.conv_out1.basisexpansion
        )
        escnn.nn.init.generalized_he_init(
            self.conv_out2.weights.data, self.conv_out2.basisexpansion
        )
        for layer in self.layers:
            layer.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, *space, channels)
        x = mv_to_gt(x, self.in_type)
        x = self.conv_in1(x)
        x = gt_to_mv(x)
        x = self.activation(x)
        x = mv_to_gt(x, self.hidden_type)
        x = self.conv_in2(x)
        x = gt_to_mv(x)
        x = self.activation(x)

        # Apply residual layers.
        for layer in self.layers:
            x = layer(x)

        x = mv_to_gt(x, self.hidden_type)
        x = self.conv_out1(x)
        x = gt_to_mv(x)
        x = self.activation(x)
        x = mv_to_gt(x, self.hidden_type)
        x = self.conv_out2(x)
        x = gt_to_mv(x)
        return x
