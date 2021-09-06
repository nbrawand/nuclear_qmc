from nuclear_qmc.operators.operators import get_sigma_ij
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_spin_isospin_indices import get_raw_spin_indices
from jax import numpy as jnp, vmap


def invert_bit(integer, nth_bit):
    return integer ^ (1 << nth_bit)


def get_flip_indices(mass_number):
    """Return matrix first index is particle being flipped, value at the 2nd index is the index of new element after
    flip."""
    indices = get_raw_spin_indices(mass_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)
    flipped_indices = vmap(lambda particle: vmap(invert_bit, in_axes=(0, None))(indices, particle))(particle_index)
    return flipped_indices  # [n_particle, n_spin_states]


def get_bit(integer, position):
    out = integer >> position & 1
    return out


def make_negative_1_if_spin_down_else_1(arr):
    arr *= 2.0
    arr -= 1.0
    return arr


def get_sigma_operator_prefactors(mass_number):
    """return prefactor for sigma_x, y, z with shapes: (1.0, [n_particles, n_indices], [n_particles, n_indices])"""
    indices = get_raw_spin_indices(mass_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)

    # sigma_x is just a flip so prefactors are 1
    sigma_x = 1.0

    # sigma_z is a negative if the original spin started down
    extracted_bits = vmap(lambda particle: vmap(get_bit, in_axes=(0, None))(indices, particle))(particle_index)
    sigma_z = make_negative_1_if_spin_down_else_1(extracted_bits)

    # sigma_y is a flip with an i and - if the spin started down
    sigma_y = 1.j * sigma_z

    return sigma_x, sigma_y, sigma_z


def sigma_i_r_ij_sigma_j_r_ij(sigma, normalized_r_ij, pair):
    i, j = pair
    sigma_i_r_ij = jnp.vdot(jnp.swapaxes(sigma[i], -1, -2), normalized_r_ij[i])
    sigma_j_r_ij = jnp.vdot(jnp.swapaxes(sigma[j], -1, -2), normalized_r_ij[j])
    return sigma_i_r_ij * sigma_j_r_ij


def get_sigma_x_i(flipped_indices, ith_particle, psi_r):
    psi_r = psi_r[:, flipped_indices[ith_particle]]
    return psi_r


def get_sigma_y_i(flipped_indices, prefactors, ith_particle, psi_r):
    psi_r *= prefactors[ith_particle]
    psi_r = psi_r[:, flipped_indices[ith_particle]]
    return psi_r


def get_sigma_z_i(prefactors, ith_particle, psi_r):
    psi_r *= prefactors[ith_particle]
    return psi_r


def get_sigma_i(flipped_indices, ith_particle, psi_r, y_prefactors, z_prefactors):
    return jnp.array([
        get_sigma_x_i(flipped_indices, ith_particle, psi_r)
        , get_sigma_y_i(flipped_indices, y_prefactors, ith_particle, psi_r)
        , get_sigma_z_i(z_prefactors, ith_particle, psi_r)
    ])


def normalize_r_ij(r_ij):
    norms = jnp.linalg.norm(r_ij, axis=1)
    norms = 1. / norms
    out = r_ij * norms[:, None]
    return out


def get_sij_psi_r(r_ij, psi_r, particle_i, particle_j, flipped_indices, y_prefactors, z_prefactors, sigma_ij):
    """Definition below equation 3 in https://arxiv.org/pdf/2001.01374.pdf
    S_{ij} = 3 \sigma_i \cdot \hat{r}_{ij} \sigma_j \cdot \hat{r}_{ij} - \sigma_{ij}

    Parameters
    ----------
    psi_r
    spin_exchange_indices
    pair_coefficients

    Returns
    -------

    """
    normalized_r_ij = r_ij / jnp.linalg.norm(r_ij)
    sigma_i = get_sigma_i(flipped_indices, particle_i, psi_r, y_prefactors, z_prefactors)
    sigma_i_r = jnp.einsum('ijk, i', sigma_i, normalized_r_ij)
    sigma_j = get_sigma_i(flipped_indices, particle_j, psi_r, y_prefactors, z_prefactors)
    sigma_j_r = jnp.einsum('ijk, i', sigma_j, normalized_r_ij)
    sij_psi_r = 3 * sigma_i_r * sigma_j_r
    sij_psi_r = sij_psi_r - sigma_ij
    return sij_psi_r
