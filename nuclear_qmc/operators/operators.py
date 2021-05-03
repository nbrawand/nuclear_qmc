import jax
import jax.numpy as jnp

from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M


def laplacian(wave_function, r_coords):
    d2_psi = jax.hessian(wave_function.psi, argnums=0)(r_coords)
    dim = r_coords.shape[0] * r_coords.shape[1]
    d2_psi = d2_psi.reshape(*wave_function.spin.shape, dim, dim)
    d2_psi = jnp.trace(d2_psi, axis1=-1, axis2=-2)
    return d2_psi


def kinetic_energy(wave_function, r_coords):
    d2_psi = laplacian(wave_function, r_coords)
    ke = - H_BAR_SQRD_OVER_2_M * jnp.vdot(wave_function.psi(r_coords), d2_psi)
    return ke


def grad(wave_function, r_coords):
    psi = lambda r, p, s: wave_function.psi_prefactor(r, p) * wave_function.psi_vector(r, p, s)
    return jax.jacfwd(psi, argnums=(1,))(r_coords, wave_function.params, wave_function.spin)


def grad_only_radial(wave_function, r_coords):
    return jax.jacfwd(wave_function.psi_prefactor, argnums=(1,))(r_coords, wave_function.params)


def tau(wave_function, r_coords, pair_coefficients):
    return _tau_or_sigma(wave_function.psi(r_coords).T, wave_function.iso_spin_exchange_indices, pair_coefficients).T


def sigma(wave_function, r_coords, pair_coefficients):
    return _tau_or_sigma(wave_function.psi(r_coords), wave_function.spin_exchange_indices, pair_coefficients)


def _tau_or_sigma(psi_r, exchange_indices, pair_coefficients):
    exchanged_psi_r = psi_r[:, exchange_indices]  # [n_isospin, n_spin, n_pair_exchanges]
    psi_r = jnp.expand_dims(psi_r, -1)  # [n_isospin, n_spin, 1]
    psi_r_prime = 2.0 * exchanged_psi_r - psi_r
    psi_r_prime *= pair_coefficients
    psi_r_prime = psi_r_prime.sum(-1)
    return psi_r_prime
