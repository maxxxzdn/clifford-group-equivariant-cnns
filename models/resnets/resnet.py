"""Adapted to JAX from https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_resnet.py"""

import jax
import math
from flax import linen as nn
from jax.nn import gelu


def xavier_uniform_init(key, shape, dtype, factor=0.1):
    """
    Xavier uniform initialization.

    Args:
        key (jax.random.PRNGKey): The random key to use for generating the multivector.
        shape (tuple): The shape of the weight matrix.
        dtype (jax.numpy.dtype): The data type of the weight matrix.
        factor (float): The scaling factor for the scheme.
            - default: 0.1, found to work best after grid search from 0.01 to 1.
    """
    bound = factor * math.sqrt(6) / math.sqrt(shape[-2] + shape[-1])
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class BasicBlock(nn.Module):
    """
    Basic block for the ResNet.

    Attributes:
        in_channels (int): The number of input channels.
        channels (int): The number of hidden channels.
        norm (bool): Whether to use normalization in the block.
        kernel_size (int): The size of the kernel.
        dim (int): The dimension of the space.
    """

    in_channels: int
    channels: int
    norm: bool
    kernel_size: int
    dim: int

    @nn.compact
    def __call__(self, x):
        """
        Applies the basic block to an input.
            x -> conv1 -> norm1 -> gelu -> conv2 -> norm2 -> x + conv(x) -> gelu -> out

        Args:
            x: The input of shape (B,T,...,C).

        Returns:
            The output of shape (B,T,...,C).
        """
        out = nn.Conv(
            features=self.channels,
            kernel_size=tuple(self.dim * [self.kernel_size]),
            kernel_init=xavier_uniform_init,
        )(x)
        out = nn.LayerNorm()(out) if self.norm else out
        out = gelu(out)
        out = nn.Conv(
            features=self.channels,
            kernel_size=tuple(self.dim * [self.kernel_size]),
            kernel_init=xavier_uniform_init,
        )(out)
        out = nn.LayerNorm()(out) if self.norm else out

        # shortcut connection
        if self.in_channels != self.channels:
            x = nn.Conv(
                features=self.channels,
                kernel_size=tuple(self.dim * [1]),
                kernel_init=xavier_uniform_init,
            )(x)
            x = nn.LayerNorm()(x) if self.norm else x

        out += x
        out = gelu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model.

    Attributes:
        time_history (int): The number of time steps in the past.
        time_future (int): The number of time steps in the future.
        hidden_channels (int): The number of hidden channels.
        kernel_size (int): The size of the kernel.
        blocks (tuple): The number of blocks in each layer.
        norm (bool): Whether to use normalization in the block.
        make_channels (bool): create channel dimension (when time is part of the grid).
    """

    time_history: int
    time_future: int
    hidden_channels: int
    kernel_size: int
    blocks: tuple = (2, 2, 2, 2)
    norm: bool = True
    make_channels: bool = False

    @nn.compact
    def __call__(self, x):
        out_dim = self.time_future if not self.make_channels else 1
        dim = len(x.shape) - 3  # input shape is (B,T,X_1, ..., X_dim,C)
        orig_shape = x.shape

        # reshape such that the time dimension is combined with the channel dimension
        x = x.transpose(0, *range(2, len(x.shape) - 1), 1, -1)  # (B,...,T,C)
        x = x.reshape(*x.shape[:-2], -1)  # (B,...,T*C)

        # Embedding convolutional layers
        x = nn.Conv(
            features=self.hidden_channels,
            kernel_size=tuple(dim * [1]),
            kernel_init=xavier_uniform_init,
        )(x)
        x = gelu(x)
        x = nn.Conv(
            features=self.hidden_channels,
            kernel_size=tuple(dim * [1]),
            kernel_init=xavier_uniform_init,
        )(x)
        x = gelu(x)

        # Basic blocks
        for num_blocks in self.blocks:
            for _ in range(num_blocks):
                x = BasicBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    norm=self.norm,
                    kernel_size=self.kernel_size,
                    dim=dim,
                )(x)

        # Output convolutional layers
        x = nn.Conv(
            features=self.hidden_channels,
            kernel_size=tuple(dim * [1]),
            kernel_init=xavier_uniform_init,
        )(x)
        x = gelu(x)
        x = nn.Conv(
            features=out_dim*orig_shape[-1],
            kernel_size=tuple(dim * [1]),
            kernel_init=xavier_uniform_init,
        )(x)

        # reshape back to original shape
        x = x.reshape(*x.shape[:-1], out_dim, orig_shape[-1])  # (B,...,T,C)
        x = x.transpose(0, -2, *range(1, len(x.shape) - 2), -1)  # (B,T,...,C)

        return x
