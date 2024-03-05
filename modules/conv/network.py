from flax import linen as nn

from ..core.fcgp import FullyConnectedSteerableGeometricProductLayer
from ..core.mvgelu import MVGELU
from ..core.norm import MVLayerNorm


class KernelNetwork(nn.Module):
    """
    Kernel network for the steerable implicit kernel.
    It uses an O(p,q)-equivariant CEGNN to generate a stack of c_in * c_out multivectors.

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
    """

    algebra: object
    c_in: int
    c_out: int
    num_layers: int
    hidden_dim: int
    bias_dims: tuple
    product_paths_sum: int

    @nn.compact
    def __call__(self, x):
        """
        Kernel network evaluation (see Appendix A for details).

        Args:
            x: The input multivector of shape (N, 1, 2**algebra.dim).

        Returns:
            The output multivector of shape (N, c_out * c_in, 2**algebra.dim).
        """
        x = FullyConnectedSteerableGeometricProductLayer(
            self.algebra,
            1,
            self.hidden_dim,
            bias_dims=self.bias_dims,
            product_paths_sum=self.product_paths_sum,
        )(x)
        x = MVLayerNorm(self.algebra)(x)
        x = MVGELU()(x)

        for _ in range(self.num_layers - 2):
            x = FullyConnectedSteerableGeometricProductLayer(
                self.algebra,
                self.hidden_dim,
                self.hidden_dim,
                bias_dims=self.bias_dims,
                product_paths_sum=self.product_paths_sum,
            )(x)
            x = MVLayerNorm(self.algebra)(x)
            x = MVGELU()(x)

        x = FullyConnectedSteerableGeometricProductLayer(
            self.algebra,
            self.hidden_dim,
            self.c_out * self.c_in,
            bias_dims=self.bias_dims,
            product_paths_sum=self.product_paths_sum,
        )(x)
        return x
