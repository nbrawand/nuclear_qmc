import jax.numpy as jnp
from jax import numpy as jnp


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


def get_r_ij_sqrd(r_coords, particle_pairs):
    """

    Parameters
    ----------
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    particle_pairs: ndarray
        [n_pairs, 2] particle indices for each pair.

    Returns
    -------
    ndarray
        [n_pairs] :math:`(r_i-r_j)^2` for each combo i<j, j in order of particle_pairs.
    """
    r_ij_sqrd = r_coords[particle_pairs[:, 0]] - r_coords[particle_pairs[:, 1]]
    r_ij_sqrd = (r_ij_sqrd ** 2).sum(axis=-1)
    return r_ij_sqrd


def get_r_ik_r_ij_sqrd(r_coords, particle_triplets, i, j, k):
    """

    Parameters
    ----------
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.
    k: int
        Particle index.
    j: int
        Particle index.
    i: int
        Particle index.

    Returns
    -------
    ndarray
        [n_triplets] :math:`(r_i-r_k)^2+(r_i-r_j)^2` in particle_triplets.
    """
    r_ik = r_coords[particle_triplets[:, i]] - r_coords[particle_triplets[:, k]]
    r_ij = r_coords[particle_triplets[:, i]] - r_coords[particle_triplets[:, j]]
    r_ik_ij = (r_ik ** 2).sum(axis=-1) + (r_ij ** 2).sum(axis=-1)
    return r_ik_ij


def get_r_ik_r_ij_cycles(r_coords, particle_triplets):
    """

    Parameters
    ----------
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.

    Returns
    -------
    ndarray
        [n_triplets] cyclic combinations of ijk of terms: :math:`(r_i-r_k)^2+(r_i-r_j)^2` in particle_triplets
    """
    cycles = get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 0, 1, 2)
    cycles = jnp.append(cycles, get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 2, 0, 1))
    cycles = jnp.append(cycles, get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 1, 2, 0))
    return cycles