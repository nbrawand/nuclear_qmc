import jax.numpy as jnp
from jax import vmap
from nuclear_qmc.operators.operators import sigma
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.utils.get_dr_ij import get_r_ij
from nuclear_qmc.wave_function.neural_network import build_nn_wfc
from nuclear_qmc.wave_function.utility import apply_confining_potential


def get_exp_rij(params, r, particle_pairs):
    r1 = r[particle_pairs[:, 0]]
    r2 = r[particle_pairs[:, 1]]
    dr = r1 - r2
    dr = jnp.linalg.norm(dr, axis=1)
    return jnp.exp(-params * dr)


def build_jastro_wave_function_with_spin_correlations(ndense, particle_pairs, spin, spin_exchange_indices):
    RuntimeError('This needs an exponential')
    """return the following wave function:
    \prod f_c_ij * (1.+\sum_ij f_sigma_ij / f_central_ij * sigma_ij) * spin"""
    n_particle_pairs = len(particle_pairs)

    # key, psi_prefactor, flat_params
    neural_network_psis = []
    neural_network_flat_params = []
    for i in range(2 * n_particle_pairs):
        _, psi, params = build_nn_wfc(ndense=ndense)
        n_params_per_network = len(params)
        neural_network_flat_params.append(params)
        neural_network_psis.append(psi)
    neural_network_flat_params = jnp.array(neural_network_flat_params).reshape(-1)

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        f_central_ij = jnp.array(
            [neural_network_psis[i](in_params[i * n_params_per_network:(1 + i) * n_params_per_network], r_coords)
             for i in range(n_particle_pairs)])
        f_spin_ij = jnp.array(
            [neural_network_psis[i](in_params[i * n_params_per_network:(1 + i) * n_params_per_network], r_coords)
             for i in range(n_particle_pairs, 2 * n_particle_pairs)])
        f_central_product = jnp.prod(f_central_ij)
        f_ratios = f_spin_ij / f_central_ij
        # \sum_ij f_sigma_ij / f_central_ij * sigma_ij
        sum_ij_sigma = sigma(lambda a, b: 1., None, spin, r_coords, spin_exchange_indices, f_ratios)
        psi = f_central_product * (spin + sum_ij_sigma)
        psi *= apply_confining_potential(r_coords)
        return psi

    return psi_function, neural_network_flat_params


def build_jastro_wave_function_no_spin_correlations_multiple_networks(ndense, particle_pairs):
    RuntimeError('This needs an exponential')
    """return the following wave function:
    \prod f_c_ij """
    n_particle_pairs = len(particle_pairs)

    # key, psi_prefactor, flat_params
    neural_network_psis = []
    neural_network_flat_params = []
    for i in range(n_particle_pairs):
        _, psi, params = build_nn_wfc(ndense=ndense)
        n_params_per_network = len(params)
        neural_network_flat_params.append(params)
        neural_network_psis.append(psi)
    neural_network_flat_params = jnp.array(neural_network_flat_params).reshape(-1)

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        f_central_ij = jnp.array(
            [neural_network_psis[i](in_params[i * n_params_per_network:(1 + i) * n_params_per_network], r_coords)
             for i in range(n_particle_pairs)])
        f_central_product = jnp.prod(f_central_ij)
        psi = f_central_product * apply_confining_potential(r_coords)
        return psi

    return psi_function, neural_network_flat_params


def build_jastro_wave_function_no_spin_correlations_single_network(key, ndense, particle_pairs):
    key, nn_func, params = build_nn_wfc(ndense=ndense, key=key)

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        dr_ij = get_r_ij(r_coords, particle_pairs)
        nn_dr_ij = vmap(nn_func, in_axes=(None, 0))(in_params, dr_ij)
        f_c_ij = jnp.exp(nn_dr_ij)
        f_c_product = jnp.prod(f_c_ij)
        psi = f_c_product * apply_confining_potential(r_coords)
        return psi

    return key, psi_function, params
