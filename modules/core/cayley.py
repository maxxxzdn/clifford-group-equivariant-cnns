import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal


class WeightedCayley(nn.Module):
    """
    Parameterized Clifford multiplication table (see Function 2 in the paper).

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        in_features (int): The number of input channels.
        out_features (int): The number of output channels.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
    """
    algebra: object
    in_features: int
    out_features: int
    product_paths_sum: int
    
    @nn.compact
    def __call__(self):
        """
        Constructs the weighted Cayley table. For each c_in-c_out pair and 
        each non-zero element in the Cayley table, a weight is initialized.

        Returns:
            jnp.ndarray: The weighted Cayley table of shape (out_features, in_features, 2**algebra.dim, 2**algebra.dim, 2**algebra.dim).
        """

        # Initializing the weights
        stddev = 1 / jnp.sqrt(self.in_features * (self.algebra.dim + 1))
        weight_init = self.param('weight', normal(stddev), 
                                 (self.out_features, self.in_features, self.product_paths_sum))

        # Non-zero elements in the Cayley table for each input-output pair
        weight = jnp.zeros((self.out_features, self.in_features) + self.algebra.geometric_product_paths.shape)
        weight = weight.at[:, :, self.algebra.geometric_product_paths].set(weight_init)

        # Repeating the weights across subspaces
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat(subspaces, axis=-3)
            .repeat(subspaces, axis=-2)
            .repeat(subspaces, axis=-1)
        )
        
        return self.algebra.cayley * weight_repeated
