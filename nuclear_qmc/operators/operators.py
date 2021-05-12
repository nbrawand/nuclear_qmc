import jax
import jax.numpy as jnp

from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M


def laplacian(psi, psi_params, r_coords):
    d2_psi = jax.hessian(psi, argnums=1)(psi_params, r_coords)
    dim1 = d2_psi.shape[-2] * d2_psi.shape[-1]
    dim2 = d2_psi.shape[-4] * d2_psi.shape[-3]
    new_dims = (*d2_psi.shape[:-4], dim1, dim2)
    d2_psi = d2_psi.reshape(*new_dims)
    d2_psi = jnp.trace(d2_psi, axis1=-1, axis2=-2)
    return d2_psi


def kinetic_energy_psi(psi, psi_params, r_coords):
    d2_psi = laplacian(psi, psi_params, r_coords)
    ke = - H_BAR_SQRD_OVER_2_M * d2_psi
    return ke


def tau(psi, psi_params, psi_vector, r_coords, iso_spin_exchange_indices, pair_coefficients):
    psi_r = psi(psi_params, r_coords) * psi_vector
    return _tau_or_sigma(psi_r.T, iso_spin_exchange_indices, pair_coefficients).T


def sigma(psi, psi_params, psi_vector, r_coords, spin_exchange_indices, pair_coefficients):
    psi_r = psi(psi_params, r_coords) * psi_vector
    return _tau_or_sigma(psi_r, spin_exchange_indices, pair_coefficients)


def _tau_or_sigma(psi_r, exchange_indices, pair_coefficients):
    exchanged_psi_r = psi_r[:, exchange_indices]  # [n_isospin, n_spin, n_pair_exchanges]
    psi_r = jnp.expand_dims(psi_r, -1)  # [n_isospin, n_spin, 1]
    psi_r_prime = 2.0 * exchanged_psi_r - psi_r
    psi_r_prime *= pair_coefficients
    psi_r_prime = psi_r_prime.sum(-1)
    return psi_r_prime
