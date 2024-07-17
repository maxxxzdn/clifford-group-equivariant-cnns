import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import ones

from ..core.cayley import WeightedCayley
from .shell import ScalarShell, compute_scalar_shell
from .network import KernelNetwork


def generate_kernel_grid(kernel_size, dim):
    """
    Generate the 2D or 3D grid for a given kernel size.

    Args:
        kernel_size (int): The size of the kernel.
        dim (int): The dimension of the grid.

    Returns:
        jnp.ndarray: The grid of shape (kernel_size ** dim, dim) defined on the range [-1, 1]^dim.
    """
    axes = [jnp.arange(0, kernel_size) for _ in range(dim)]
    grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    grid = grid - kernel_size // 2
    return grid.reshape(-1, dim) / max(kernel_size // 2, 1.0)


def get_init_factor(algebra, kernel_size):
    """
    Compute initial factor for the kernel (empirical).

    Args:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        kernel_size (int): The size of the kernel.

    Returns:
        float: The initial factor for the kernel.
    """
    return 20 / kernel_size ** (algebra.dim - 1)


class CliffordSteerableKernel(nn.Module):
    """
    Clifford-steerable kernel (see Section 3, Appendix A for details). Pseudocode is given in Function 2.
    It consists of two components:
        1. A kernel network that generates a stack of c_in * c_out multivectors using an O(p,q)-equivariant CEGNN (Ruhe et al., 2023)
            k: R^p,q -> Cl^(c_out * c_in)
        2. A kernel head that converts the stack of multivectors into a kernel.
            K: Cl^[c_out x c_in] -> Hom_vec(Cl^c_in, Cl^c_out)

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    num_layers: int
    hidden_dim: int
    bias_dims: tuple
    product_paths_sum: int

    def setup(self):
        self.rel_pos = generate_kernel_grid(self.kernel_size, self.algebra.dim)[
            :, jnp.newaxis, :
        ]
        self.factor = get_init_factor(self.algebra, self.kernel_size)
        self.rel_pos_sigma = self.param("rel_pos_sigma", ones, (1, 1, 1))

    @nn.compact
    def __call__(self):
        """
        Evaluate the steerable implicit kernel.

        Returns:
            The output kernel of shape (c_out * algebra.n_blades, c_in * algebra.n_blades, X_1, ..., X_dim).
        """
        # Weighted Cayley
        weighted_cayley = WeightedCayley(
            self.algebra, self.c_in, self.c_out, self.product_paths_sum
        )()

        # Compute scalars
        scalar = compute_scalar_shell(self.algebra, self.rel_pos, self.rel_pos_sigma)

        # Embed scalar and vector into a multivector
        x = self.algebra.embed_grade(scalar, 0) + self.algebra.embed_grade(
            self.rel_pos, 1
        )

        # Evaluate kernel network
        k = KernelNetwork(
            self.algebra,
            self.c_in,
            self.c_out,
            self.num_layers,
            self.hidden_dim,
            self.bias_dims,
            self.product_paths_sum,
        )(x)

        # Reshape to kernel mask
        k = k.reshape(-1, self.c_out, self.c_in, self.algebra.n_blades)

        # Compute kernel mask
        shell = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2**self.algebra.dim
        )

        # Mask kernel
        k = k * shell * self.factor

        # Kernel head: partial weighted geometric product
        K = jnp.einsum("noik,oiklm->olimn", k, weighted_cayley)

        # Reshape to final kernel
        K = K.reshape(
            self.c_out * self.algebra.n_blades,
            self.c_in * self.algebra.n_blades,
            *(self.algebra.dim * [self.kernel_size])
        )

        return K
