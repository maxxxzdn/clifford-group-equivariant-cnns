""" Inspired by https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/nonlinearities.py"""

import jax
import jax.numpy as jnp
from flax import linen as nn


class MVGELU(nn.Module):

    def __call__(self, input):
        """
        Multivector adaptation of the GELU activation function.
        GELU is computed for the scalar part of the input and used as a gate for the multivector input:

            out = GELU(input[0]) * input

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            out (jnp.ndarray): multivector of shape (..., 2**algebra.dim).
        """
        scalar = input[..., [0]]
        gates = jax.nn.sigmoid(
            jnp.sqrt(2 / jnp.pi) * (2 * (scalar + 0.044715 * jnp.power(scalar, 3)))
        )
        return gates * input
