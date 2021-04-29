from itertools import combinations
import jax.numpy as jnp


def get_triplets(int_arr):
    """
    Get an array of all triplet combinations (ijk) of elements in int_arr such that (i<j<k)

    Parameters
    ----------
    int_arr: ndarray 1D array of reals

    Returns
    -------
    ndarray[n_combindations, 3]
    """
    triplets = jnp.array(list(combinations(int_arr, 3)))
    triplets = triplets[jnp.where(triplets[:, 0] < triplets[:, 1])]
    triplets = triplets[jnp.where(triplets[:, 1] < triplets[:, 2])]
    return triplets
