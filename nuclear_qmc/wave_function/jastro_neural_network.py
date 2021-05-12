import jax.numpy as jnp
from nuclear_qmc.operators.operators import sigma
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.wave_function.neural_network import build_nn_wfc


def get_exp_rij(params, r, particle_pairs):
    r1 = r[particle_pairs[:, 0]]
    r2 = r[particle_pairs[:, 1]]
    dr = r1 - r2
    dr = jnp.linalg.norm(dr, axis=1)
    return jnp.exp(-params * dr)


def build_jastro_wave_function_with_spin_correlations(ndense, particle_pairs, spin, spin_exchange_indices):
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
        return psi

    return psi_function, neural_network_flat_params
