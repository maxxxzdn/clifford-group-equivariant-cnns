from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import uniform


def compute_scalar_shell(algebra: object, v: jnp.ndarray, sigma: jnp.ndarray):
    """
    Compute scalar shell for the output of the kernel network given a vector.
    
    Args:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        v (jnp.ndarray): The input vector of shape (N, 1, algebra.dim).
        sigma (jnp.ndarray): The array of kernel widths.
            - last dimension can be 2**algebra.dim or 1.
        
    Returns:
        jnp.ndarray: The output scalar of shape (N, 1, 1).
    """
    q_v = algebra.q(algebra.embed_grade(v,1))
    sgn = jnp.where(q_v >= 0, 1, -1)
    return sgn * jnp.exp(-jnp.abs(q_v) / (2 * sigma ** 2))


class ScalarShell(nn.Module):
    """
    Kernel mask used inside steerable implicit kernels.
    See Appendix A for details, pseudocode for a single channel is given in Function 1.

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
    """
    algebra: object
    c_in: int
    c_out: int
    
    @nn.compact
    def __call__(self, x):
        """
        Compute scalar shell for the output of the kernel network.

        Args:
            x: The input multivector of shape (N, 1, 2**algebra.dim).

        Returns:
            The output multivector of shape (N, c_out * c_in, 2**algebra.dim).
        """
        kernel_width = self.param(
            'kernel_width',
            uniform(scale=0.2),
            (self.c_out, self.c_in, self.algebra.n_subspaces)
        ) + 0.4
        
        # broadcasting along the grades
        kernel_width = jnp.repeat(kernel_width, self.algebra.subspaces, axis=-1)
        kernel_width = kernel_width.reshape(1, -1, 2 ** self.algebra.dim)
        return compute_scalar_shell(self.algebra, x, kernel_width)