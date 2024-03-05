"""Adapted to JAX from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/algebra/metric.py"""

import functools
import math
import jax
import jax.numpy as jnp

from .metric import ShortLexBasisBladeOrder, construct_gmt


def _smooth_abs_sqrt(input, eps=1e-16, min_clamp=0.01):
    """
    Computes a smooth approximation of the absolute value of the square root.

    Args:
        input: The input value.
        eps (float, optional): A small epsilon value for numerical stability.
        min_clamp (float, optional): small numerical factor to avoid instabilities
            - inspired by https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/normalization.py

    Returns:
        The smooth absolute square root of the input.
    """
    input = jnp.clip(input, min_clamp, None)
    return (input**2 + eps) ** 0.25


class CliffordAlgebra:
    def __init__(self, metric: jnp.array):
        """
        Represents a Clifford Algebra defined by a given metric.

        Attributes:
            metric (jnp.array): The metric tensor for the algebra.
            num_bases (int): Number of basis vectors in the algebra.
            bbo (ShortLexBasisBladeOrder): Basis blade order for the algebra.
            dim (int): Dimension of the algebra.
            n_blades (int): Total number of blades in the algebra.
            grades (list): Unique grades present in the algebra.
            subspaces (jnp.array): Array of subspace dimensions.
            n_subspaces (int): Number of subspaces in the algebra.
            grade_to_slice (list): Mapping from grades to slices in the algebra.
            grade_to_index (list): Mapping from grades to indices.
            bbo_grades (jnp.array): Grades of the basis blades.
            even_grades (jnp.array): Indicator array for even grades.
            odd_grades (jnp.array): Indicator array for odd grades.
            cayley (jnp.array): Cayley table for the algebra.
            geometric_product_paths (jnp.array): Paths in the geometric product.
            geometric_product_paths_sum (int): Sum of the geometric product paths.
        """
        self.metric = jnp.array(metric, dtype=jnp.int32)
        self.num_bases = len(self.metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        cayley = construct_gmt(
            self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric
        ).astype(jnp.float32)

        self.grades = jnp.unique(self.bbo.grades).tolist()
        self.subspaces = jnp.array([math.comb(self.dim, g) for g in self.grades])
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        self.grade_to_index = [
            jnp.arange(*s.indices(s.stop)) for s in self.grade_to_slice
        ]

        self.bbo_grades = self.bbo.grades
        self.even_grades = self.bbo_grades % 2 == 0
        self.odd_grades = ~self.even_grades
        self.cayley = cayley
        self.geometric_product_paths = self._calculate_geometric_product_paths()
        self.geometric_product_paths_sum = self.geometric_product_paths.sum().item()

    @property
    def _beta_signs(self):
        """
        Computes the signs for the main anti-involution.

        Returns:
            Signs for the main anti-involution.
        """
        if not hasattr(self, "__beta_signs_cache"):
            self.__beta_signs_cache = jnp.power(
                -1, self.bbo_grades * (self.bbo_grades - 1) // 2
            )
        return self.__beta_signs_cache

    def _grade_to_slice(self, subspaces):
        """
        Converts subspaces into slices for indexing.

        Args:
            subspaces (jnp.array): Array of subspace dimensions.

        Returns:
            List of slice objects corresponding to each grade.
        """
        grade_to_slice = list()
        subspaces = jnp.array(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    def _calculate_geometric_product_paths(self):
        """
        Calculates the paths in the geometric product for the algebra.

        Returns:
            A 3D boolean array indicating the presence of paths in the geometric product.
        """
        gp_paths = jnp.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths = gp_paths.at[i, j, k].set((m != 0).any())

        return gp_paths

    @functools.partial(jax.jit, static_argnums=(0,))
    def geometric_product(self, a, b, blades=None):
        """
        Computes the geometric product of two multivectors.

        Args:
            a, b: Multivectors to be multiplied.
            blades (optional): Specific blades to consider in the product.

        Returns:
            The geometric product of multivector `a` and multivector `b`.
        """
        cayley = self.cayley

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        return jnp.einsum("...i,ijk,...k->...j", a, cayley, b)

    def exponential(self, mv, truncate=8):
        """
        Computes the exponential of a multivector.

        Args:
            mv: The multivector to compute the exponential of.
            truncate (int, optional): The number of terms to include in the series expansion.

        Returns:
            The exponential of the multivector.
        """
        x = mv
        result = mv.clone()
        result = result.at[..., 0].add(1)
        for i in range(2, truncate):
            x = self.geometric_product(x, mv)
            result += x / math.factorial(i)
        return result

    def sandwich(self, u, v, w):
        """
        Sandwich product of three multivectors.

        Args:
            u, v, w: multivectors.

        Returns:
            (u * v) * w
        """
        return self.geometric_product(self.geometric_product(u, v), w)

    def embed(self, array, array_index):
        """
        Embeds an array into a multivector at a specified index.

        Args:
            array (jnp.array): The array to be embedded.
            array_index (int): The index at which to embed the array.

        Returns:
            A multivector with the given array embedded at the specified index.
        """
        indices = (..., array_index)
        mv = (
            jnp.zeros((*array.shape[:-1], 2**self.dim), dtype=array.dtype)
            .at[indices]
            .set(array)
        )
        return mv

    def embed_grade(self, array, grade):
        """
        Embeds an array into a multivector at a specified grade.

        Args:
            array (jnp.array): The array to be embedded.
            grade (int): The grade at which to embed the array.

        Returns:
            A multivector with the given array embedded at the specified grade.
        """
        return (
            jnp.zeros((*array.shape[:-1], 2**self.dim))
            .at[..., self.grade_to_index[grade]]
            .set(array)
        )

    def get_grade(self, mv, grade):
        """
        Extracts the components of a multivector that belong to a specified grade.

        Args:
            mv: The multivector from which to extract components.
            grade (int): The grade of the components to extract.

        Returns:
            A multivector containing only the components of the specified grade.
        """
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def random_grade(self, key, grade: int, n=None):
        """
        Generates a random multivector where a specific grade is non-zero.

        Args:
            key (jax.random.PRNGKey): The random key to use for generating the multivector.
            grade (int): The grade of the multivector to generate.
            n (optional): The number of multivectors to generate.

        Returns:
            A random multivector of the specified grade.
        """
        if n is None:
            n = 1
        grade_indices = self.bbo_grades == grade
        mv = jnp.zeros((n, self.n_blades))
        components = jax.random.normal(key, (n, grade_indices.sum()))
        mv = mv.at[..., grade_indices].set(components)
        return mv

    def beta(self, mv, blades=None):
        """
        Computes the main anti-involution (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            mv: The multivector to compute the anti-involution of.
            blades (optional): Specific blades to consider in the product.

        Returns:
            Main anti-involution of the multivector.
        """
        signs = self._beta_signs[blades] if blades is not None else self._beta_signs
        return signs * mv

    def b(self, x, y, blades=None):
        """
        Computes the extended bilinear form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            x, y: Multivectors to be multiplied.
            blades (optional): Specific blades to consider in the product.

        Returns:
            Projection onto the zero-component of beta(x, y).
        """
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                jnp.array([0]),
                blades[1],
            )
        else:
            blades = jnp.arange(self.n_blades)
            blades = (
                blades,
                jnp.array([0]),
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        """
        Computes the extended quadratic form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            mv: The multivector to compute the quadratic form of.
            blades (optional): Specific blades to consider in the product.

        Returns:
            b(mv, mv).
        """
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def qs(self, mv, grades=None):
        """
        Computes the quadratic forms of a multivector for specific grades.

        Args:
            mv: The multivector to compute the quadratic forms of.
            grades (optional): Specific grades to consider in the computation.

        Returns:
            A list of quadratic forms for each specified grade.
        """
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def eta(self, w):
        """
        Coboundary of Clifford main involution (see Eq. 361 of https://arxiv.org/abs/2305.11141)

        Args:
            w: The multivector to compute the eta function of.

        Returns:
            -1 if the multivector is odd, 1 if it is even.
        """
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        """
        Clifford main involution (see Eq. 369 of https://arxiv.org/abs/2305.11141)

        Args:
            w: element of the Clifford group.
            mv: The multivector to apply the transformation to.

        Returns:
            The transformed multivector.
        """
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def norm(self, mv, blades=None):
        """
        Computes the norm of a multivector.

        Args:
            mv: The multivector to compute the norm of.
            blades (optional): Specific blades to consider in the computation.

        Returns:
            The norm of the multivector.
        """
        return _smooth_abs_sqrt(self.q(mv, blades=blades))

    def norms(self, mv, grades=None):
        """
        Computes the norms of a multivector for specific grades.

        Args:
            mv: The multivector to compute the norms of.
            grades (optional): Specific grades to consider in the computation.

        Returns:
            A list of norms for each specified grade.
        """
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def parity(self, mv):
        """
        Determines the parity (even or odd) of a multivector.

        Args:
            mv: The multivector to determine the parity of.

        Returns:
            True if the multivector is odd, False if it is even.

        Raises:
            ValueError: If the multivector is not a homogeneous element.
        """
        is_odd = jnp.allclose(mv[..., self.even_grades], 0)
        is_even = jnp.allclose(mv[..., self.odd_grades], 0)
        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def inverse(self, mv, blades=None):
        """
        Computes the inverse of a multivector.

        Args:
            mv: The multivector to compute the inverse of.
            blades (optional): Specific blades to consider in the computation.

        Returns:
            The inverse of the multivector.
        """
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def _rho_(self, w, mv):
        """
        Applies the action w to a multivector.

        Args:
            w: element of the Clifford group.
            mv: The multivector to apply the action w to.

        Returns:
            Action of w on mv.
        """
        return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))
