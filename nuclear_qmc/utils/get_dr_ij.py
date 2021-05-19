import jax.numpy as jnp


def get_dr_ij(r_coords, particle_pairs):
    """Get coordinate differences between each particle pair.

    Parameters
    ----------
    r_coords: ndarray
        [n_particles, n_coords] particle coordinates.
    particle_pairs: ndarray
        [n_pairs, 2] particle pair indices.

    Returns
    -------
    ndarray
        [n_dimensions]:math:`\\delta r` for each particle pair.

    """
    return r_coords[particle_pairs[:, 0]] - r_coords[particle_pairs[:, 1]]


def get_r_ij(r_coords, particle_pairs):
    """

    Parameters
    ----------
    r_coords: ndarray [n_particles, n_coords]
    particle_pairs: ndarray [n_pairs, 2]

    Returns
    -------
    \\delta r for each particle pair

    """
    if r_coords.ndim != 2:
        raise RuntimeError('get_r_ij requires r_coords to be a ndarray of [n_particles, n_dimensions]')
    if particle_pairs.ndim != 2:
        raise RuntimeError('get_r_ij requires particle_pairs to be a ndarray of [n_pairs, 2]')
    dr_ij = get_dr_ij(r_coords, particle_pairs)
    r_ij = jnp.linalg.norm(dr_ij, axis=1)
    return r_ij
