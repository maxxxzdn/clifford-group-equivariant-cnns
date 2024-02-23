import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros


class MVLayerNorm(nn.Module):
    """
    Equivariant layer normalization of multivectors.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
    """
    algebra: object

    def __call__(self, input):
        """Compute norm of the input and normalize the input as input / norm.
        The norm is computed w.r.t the extended quadratic form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            output (jnp.ndarray): normalized multivector of shape (..., 2**algebra.dim).
        """
        norms = self.algebra.norm(input).mean(axis=1, keepdims=True)
        outputs = input / (norms + 1e-6)
        return outputs


class GradeNorm(nn.Module):
    """
    Equivariant per-grade normalization of multivectors with learnable factors.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
    """
    algebra: object

    @nn.compact
    def __call__(self, input):
        """Compute norms of the input grades and normalize the input as input / (factor * norms).
        The norms are computed w.r.t the extended quadratic form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            output (jnp.ndarray): normalized multivector of shape (..., 2**algebra.dim).
        """        
        norms = jnp.concatenate(self.algebra.norms(input), axis=-1)
        factor = self.param('factor', zeros, (1, norms.shape[1], self.algebra.n_subspaces)) 
        factor = jax.lax.broadcast_in_dim(factor, norms.shape, (0, 1, len(norms.shape) - 1))
        norms = jnp.repeat(jax.nn.sigmoid(factor) * (norms - 1) + 1, self.algebra.subspaces, axis=-1)
        return input / (norms + 1e-6)