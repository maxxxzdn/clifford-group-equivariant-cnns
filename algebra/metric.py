"""Adapted to JAX from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/algebra/metric.py"""

import functools
import itertools
import operator

import jax.numpy as jnp


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class ShortLexBasisBladeOrder:
    """
    Class to define and store the order of basis blades in Clifford algebra.
    This ordering is based on the "short lexicographical" order of the basis vectors.
    """

    def __init__(self, n_vectors):
        # Preallocate arrays to store mappings and grades.
        self.index_to_bitmap = jnp.empty(2**n_vectors, dtype=jnp.int32)
        self.grades = jnp.empty(2**n_vectors, dtype=jnp.int32)
        self.bitmap_to_index = jnp.empty(2**n_vectors, dtype=jnp.int32)

        # Iterate through the powerset to populate the mappings and grade arrays.
        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap = self.index_to_bitmap.at[i].set(bitmap)
            self.grades = self.grades.at[i].set(len(t))
            self.bitmap_to_index = self.bitmap_to_index.at[bitmap].set(i)


def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in `x`, in ascending order"""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


def count_set_bits(bitmap: int) -> int:
    """Counts the number of bits set to 1 in bitmap"""
    count = 0
    for _ in set_bit_indices(bitmap):
        count += 1
    return count


def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign


def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
    """
    Constructs Clifford multiplication table for a given Clifford algebra.
    For definition, see Eq. 51 in Appendix A.
    The table is represented as a 2^dim x 2^dim x 2^dim matrix, where each entry contains the sign of the product
    of two basis blades. The signature of the algebra determines the metric used.
    """
    n = len(index_to_bitmap)
    gmt_matrix = jnp.zeros((n, n, n), dtype=jnp.int32)

    for i in range(n):
        bitmap_i = index_to_bitmap[i]

        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]

            gmt_matrix = gmt_matrix.at[i, v, j].set(mul)

    return gmt_matrix
