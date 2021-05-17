import jax.numpy as jnp


def reorder_from_idx(idx, a):
    return jnp.concatenate((a[idx:], a[:idx]))


def get_cyclic_permutations(array):
    return jnp.array([reorder_from_idx(i, array) for i in range(len(array))])
