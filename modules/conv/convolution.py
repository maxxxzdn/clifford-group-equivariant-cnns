import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros

from .kernel import CliffordSteerableKernel


class CliffordSteerableConv(nn.Module):
    """
    Clifford-steerable convolution layer. See Section 3, Appendix A for details. Pseudocode is given in Function 3.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        padding (bool): Whether to use padding in the convolution.
        stride (int): The stride of the convolution.
        bias (bool): Whether to use bias in the convolution.
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        # Initializing kernel
        kernel = CliffordSteerableKernel(
            algebra=self.algebra,
            c_in=self.c_in,
            c_out=self.c_out,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            bias_dims=self.bias_dims,
            product_paths_sum=self.product_paths_sum,
        )()

        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (1, self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
            )
            bias = self.algebra.embed(bias_param, self.bias_dims)

        # Reshaping multivector input for compatibiltiy with jax.lax.conv:
        # (N, c_in, X_1, ..., X_dim, 2**algebra.dim) -> (N, c_in * 2**algebra.dim, X_1, ..., X_dim)
        batch_size, input_channels = x.shape[0], self.c_in * self.algebra.n_blades
        spatial_dims = x.shape[-(self.algebra.dim + 1) : -1]
        inputs = (
            jnp.transpose(x, (0, 1, 4, 2, 3))
            if self.algebra.dim == 2
            else jnp.transpose(x, (0, 1, 5, 2, 3, 4))
        )
        inputs = inputs.reshape(batch_size, input_channels, *spatial_dims)

        # Determine padding
        if self.padding_mode.upper() == "SAME":
            padding = "SAME"
        elif self.padding:
            padding = "VALID"
            padding_size = [(self.kernel_size - 1) // 2] * self.algebra.dim
            inputs = jnp.pad(
                inputs,
                [(0, 0), (0, 0)] + [(p, p) for p in padding_size],
                mode=self.padding_mode,
            )
        else:
            padding = "VALID"

        # Convolution
        output = jax.lax.conv(
            inputs,
            kernel,
            window_strides=(self.stride,) * self.algebra.dim,
            padding=padding,
        )

        # Reshaping back to multivector
        output = output.reshape(
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )
        output = (
            jnp.transpose(output, (0, 1, 3, 4, 2))
            if self.algebra.dim == 2
            else jnp.transpose(output, (0, 1, 3, 4, 5, 2))
        )

        if self.bias:
            output = output + bias

        return output
