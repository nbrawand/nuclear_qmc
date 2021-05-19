import jax.numpy as jnp


def reorder_from_idx(idx, a):
    """Reorder elements of array `a` at index `idx`.

    Parameters
    ----------
    idx: int
        Index to reorder elements around.
    a: ndarray
        1D array.

    Returns
    -------
    ndarray

    """
    return jnp.concatenate((a[idx:], a[:idx]))


def get_cyclic_permutations(array):
    """Get all cyclic permutations of array elements.

    Parameters
    ----------
    array: ndarray
        Array to return permutations of.

    Returns
    -------
    ndarray
        [n_permutations, n_elements]

    """
    return jnp.array([reorder_from_idx(i, array) for i in range(len(array))])
