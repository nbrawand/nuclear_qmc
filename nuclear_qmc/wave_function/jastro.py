import jax.numpy as jnp
from nuclear_qmc.operators.operators import sigma
from nuclear_qmc.utils.center_particles import center_particles


def get_exp_rij(params, r, particle_pairs):
    r1 = r[particle_pairs[:, 0]]
    r2 = r[particle_pairs[:, 1]]
    dr = r1 - r2
    dr = jnp.linalg.norm(dr, axis=1)
    return jnp.exp(-params * dr)


def build_jastro_wave_function(particle_pairs):
    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        exp_rij = get_exp_rij(in_params, r_coords, particle_pairs)
        psi = jnp.prod(exp_rij)
        return psi

    return psi_function


def build_jastro_wave_function_with_spin_correlations(particle_pairs, spin, spin_exchange_indices):
    """return the following wave function:
    \prod f_c_ij * (1.+\sum_ij f_sigma_ij / f_central_ij * sigma_ij) * spin"""
    n_particle_pairs = len(particle_pairs)

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        central_params = in_params[:n_particle_pairs]
        f_central_ij = get_exp_rij(central_params, r_coords, particle_pairs)
        spin_correlation_params = in_params[n_particle_pairs:2 * n_particle_pairs]
        f_spin_ij = get_exp_rij(spin_correlation_params, r_coords, particle_pairs)
        f_central_product = jnp.prod(f_central_ij)
        f_ratios = f_spin_ij / f_central_ij
        # \sum_ij f_sigma_ij / f_central_ij * sigma_ij
        sum_ij_sigma = sigma(lambda a, b: 1., None, spin, r_coords, spin_exchange_indices, f_ratios)
        psi = f_central_product * (spin + sum_ij_sigma)
        return psi

    return psi_function
