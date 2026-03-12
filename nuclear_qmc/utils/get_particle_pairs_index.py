import jax.numpy as jnp
import numpy as np
from jax import vmap


def get_particle_pairs_index(particle_pairs):
    """
        [[0, 1],
        [0, 2],
        [1, 2]] ->

        [0, 0,    1]
        [0, 0, 2]
        [0, 0, 0]
        a[0, 1]=> 0,
        a[0, 2]=> 1,
        a[1, 2]=> 2


    Parameters
    ----------
    particle_pairs: ndarray

    Returns
    -------

    """
    max_dim = particle_pairs.max() + 1
    pairs_index = np.zeros(shape=(max_dim, max_dim))
    for i, p in enumerate(particle_pairs): pairs_index[p[0], p[1]] = i
    for i, p in enumerate(particle_pairs): pairs_index[p[1], p[0]] = i
    return jnp.array(pairs_index, dtype=jnp.int32)
