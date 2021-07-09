import jax.numpy as jnp
from nuclear_qmc.utils.get_cyclic_permutations import get_cyclic_permutations
from jax import vmap, numpy as jnp
from nuclear_qmc.operators.operators import sigma, sigma_psi_r, tau_psi_r, sigma_tau_psi_r
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.utils.get_dr_ij import get_r_ij
from nuclear_qmc.utils.get_particle_pairs_index import get_particle_pairs_index
from operator import add


def build_2b_jastro(func_2b, particle_pairs, include_distance_in_2b):
    def psi_function(in_params, in_r_coords):
        x = get_r_ij(in_r_coords, particle_pairs)
        if include_distance_in_2b:
            r_mag = vmap(jnp.linalg.norm)(in_r_coords)
            r_i_mag = r_mag[particle_pairs[:, 0]]
            r_j_mag = r_mag[particle_pairs[:, 1]]
            r_i_mag_plus_r_j_mag = vmap(add)(r_i_mag, r_j_mag)
            x = jnp.column_stack((x, r_i_mag_plus_r_j_mag))
        f_2b_ij = vmap(func_2b, in_axes=(None, 0))(in_params, x)
        psi = jnp.prod(f_2b_ij)
        return psi

    return psi_function


def build_3b_jastro(func_3b, particle_pairs, particle_triplets):
    triplet_cycles_for_all_triplets = vmap(get_cyclic_permutations)(
        particle_triplets)  # dims = [n_particle_triplets, 4, 3]
    pairs_index = get_particle_pairs_index(particle_pairs)

    def f_ij_f_jk(f, triplet):
        i, j, k = triplet
        return f[pairs_index[i, j]] * f[pairs_index[j, k]]

    def one_minus_sum_f_ij_f_jk(f, triplet_cycles):
        terms = vmap(lambda triplet: f_ij_f_jk(f, triplet))(triplet_cycles)
        result = jnp.sum(terms)
        return 1.0 - result

    def psi_function(in_params, in_r_coords):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_3b_ij = vmap(func_3b, in_axes=(None, 0))(in_params, r_ij)
        three_b_factors = vmap(one_minus_sum_f_ij_f_jk, in_axes=(None, 0))(f_3b_ij, triplet_cycles_for_all_triplets)
        psi = jnp.prod(three_b_factors)
        return psi

    return psi_function


def build_sigma_jastro(func_s, particle_pairs, spin_exchange_indices):
    """Returns the spin-isospin vector. Set psi_vector for calculation to 1.0.

    Parameters
    ----------
    func_s
    particle_pairs
    spin
    spin_exchange_indices
    func_2b

    Returns
    -------

    """

    def psi_function(in_params, in_r_coords, psi):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_s_ij = vmap(func_s, in_axes=(None, 0))(in_params, r_ij)
        sum_ij_sigma = sigma_psi_r(psi, spin_exchange_indices, f_s_ij)
        return sum_ij_sigma

    return psi_function


def build_tau_jastro(func_tau, particle_pairs, isospin_exchange_indices):
    """Returns the spin-isospin vector. Set psi_vector for calculation to 1.0.

    Parameters
    ----------
    func_tau
    particle_pairs
    spin
    spin_exchange_indices
    func_2b

    Returns
    -------

    """

    def psi_function(in_params, in_r_coords, psi):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_tau_ij = vmap(func_tau, in_axes=(None, 0))(in_params, r_ij)
        sum_ij_tau = tau_psi_r(psi, isospin_exchange_indices, f_tau_ij)
        return sum_ij_tau

    return psi_function


def build_sigma_tau_jastro(func_sigma_tau, particle_pairs, spin_exchange_indices,
                           isospin_exchange_indices):
    """Returns the spin-isospin vector. Set psi_vector for calculation to 1.0.

    Parameters
    ----------
    func_sigma_tau
    particle_pairs
    spin
    spin_exchange_indices
    func_2b

    Returns
    -------

    """

    def psi_function(in_params, in_r_coords, psi):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_ij = vmap(func_sigma_tau, in_axes=(None, 0))(in_params, r_ij)
        sum_ij_tau = sigma_tau_psi_r(psi, spin_exchange_indices, isospin_exchange_indices, f_ij)
        return sum_ij_tau

    return psi_function


def exponential_jastro(params, r_coords):
    rcm = jnp.mean(r_coords, axis=0)
    r = r_coords - rcm[None, :]
    n_particles = r.shape[0]
    delta_r = 0
    for i in range(n_particles):
        for j in range(i):
            delta_r += jnp.linalg.norm(r[i, :] - r[j, :])
    return jnp.exp(- delta_r / params[0])
