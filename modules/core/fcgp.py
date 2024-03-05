""" Adapted to JAX from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/modules/fcgp.py"""

import jax.numpy as jnp
from flax import linen as nn

from .norm import GradeNorm
from .linear import MVLinear
from .cayley import WeightedCayley

from typing import Tuple


class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    """
    Fully connected layer using steerable geometric products in Clifford algebra.

    This layer combines equivariant linear transformations and weighted geometric product.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias_dims (Tuple[int, ...]): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
        include_first_order (bool): Whether to compute the linear term in the output. Default is False.
        normalization (bool): Whether to apply grade-wise normalization to the inputs. Default is True.
    """

    algebra: object
    in_features: int
    out_features: int
    bias_dims: Tuple[int, ...]
    product_paths_sum: int
    include_first_order: bool = True
    normalization: bool = True

    @nn.compact
    def __call__(self, input):
        """
        Defines the computation performed at every call.

        Args:
            input: The input tensor to the layer.

        Returns:
            The output tensor of the fully connected steerable geometric product layer.
        """

        # Creating a weighted Cayley transform
        weighted_cayley = WeightedCayley(
            self.algebra, self.in_features, self.out_features, self.product_paths_sum
        )()

        # Applying a linear transformation to the input
        input_right = MVLinear(
            self.algebra, self.in_features, self.in_features, bias_dims=None
        )(input)

        if self.normalization:
            input_right = GradeNorm(self.algebra)(input_right)

        # Weighted geometric product
        out = jnp.einsum(
            "bn...i, mnijk, bn...k -> bm...j", input, weighted_cayley, input_right
        )

        if self.include_first_order:
            # Adding the linear term
            out += MVLinear(
                self.algebra,
                self.in_features,
                self.out_features,
                bias_dims=self.bias_dims,
            )(input)
            out = out / jnp.sqrt(2)

        return out
