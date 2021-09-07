"""Definitions from https://arxiv.org/pdf/2001.01374.pdf"""

from nuclear_qmc.operators.tensor_forces import get_sigma_i
from jax import jacfwd, vmap
import jax.numpy as jnp


def get_orbit_angular_momentum_L_ij(psi, psi_params, r_coords, particle_pairs, r_ij):
    """\frac{1}{2i} r_ij \cross (\nabla_i - \nabla_j)"""
    my_psi = lambda r: psi(psi_params, r)
    grad_psi = jacfwd(my_psi)(r_coords)  # [n_psi_iso, n_psi_spin, n_particle, n_dim]
    d_grad_psi = grad_psi[:, :, particle_pairs[:, 0]] - grad_psi[:, :,
                                                        particle_pairs[:, 1]]  # [n_iso, n_spin, n_pair, n_dim]
    L = vmap(lambda a, b: jnp.cross(a, b, axisa=0, axisb=-1), in_axes=(0, 2))(r_ij,
                                                                              d_grad_psi)  # [n_pair, n_iso, n_spin, n_dim]
    L /= 2.j
    return L


def get_total_spin_Sij(psi, psi_params, r_coords, flipped_indices, y_prefactors, z_prefactors, particle_pairs):
    """(\sigma_i + \sigma_j)/2"""
    psi_r = psi(psi_params, r_coords)
    sigma_i = vmap(get_sigma_i, in_axes=(None, 0, None, None, None))(flipped_indices
                                                                     , jnp.arange(len(r_coords))
                                                                     , psi_r, y_prefactors,
                                                                     z_prefactors)  # [n_particles, n_dims, n_iso, n_spin]
    total_spin = sigma_i[particle_pairs[:, 0]] + sigma_i[particle_pairs[:, 1]]  # [n_pair, n_dims, n_iso, n_spin]
    total_spin /= 2.0
    total_spin = jnp.moveaxis(total_spin, 1, -1)  # [n_pair, n_iso, n_spin, n_dim]
    return total_spin


def spin_orbit_operator(psi, psi_params, r_coords, r_ij, flipped_indices, y_prefactors, z_prefactors, particle_pairs):
    """L \cdot S"""
    L = get_orbit_angular_momentum_L_ij(psi, psi_params, r_coords, particle_pairs, r_ij)
    S = get_total_spin_Sij(psi, psi_params, r_coords, flipped_indices, y_prefactors, z_prefactors, particle_pairs)
    out = jnp.einsum('ijkl, ijkl-> ijk', L, S)  # [n_pair, n_iso, n_spin]
    return out
