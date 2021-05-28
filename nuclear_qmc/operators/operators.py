import jax
from jax import vmap
from nuclear_qmc.wave_function.utility import get_psi_r
import jax.numpy as jnp
from functools import partial
from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M


@partial(jax.jit, static_argnums=(0,))
def laplacian(psi, psi_params, r_coords):
    """

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.

    Returns
    -------
    ndarray
        :math:`\\nabla^2 \\Psi_s(R)`

    """
    d2_psi = jax.hessian(psi, argnums=1)(psi_params, r_coords)
    dim1 = d2_psi.shape[-2] * d2_psi.shape[-1]
    dim2 = d2_psi.shape[-4] * d2_psi.shape[-3]
    new_dims = (*d2_psi.shape[:-4], dim1, dim2)
    d2_psi = d2_psi.reshape(*new_dims)
    d2_psi = jnp.trace(d2_psi, axis1=-1, axis2=-2)
    return d2_psi


def kinetic_energy_psi(psi, psi_params, r_coords):
    """

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.

    Returns
    -------
    float
        Kinetic energy.

    """
    d2_psi = laplacian(psi, psi_params, r_coords)
    ke = - H_BAR_SQRD_OVER_2_M * d2_psi
    return ke


def tau(psi, psi_params, psi_vector, r_coords, iso_spin_exchange_indices, pair_coefficients):
    """Evaluate sum_ij tau_ij on psi_r first evaluate psi_r using vector and psi

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    iso_spin_exchange_indices: ndarray
        2D array containing the indices after applying :math:`\\tau_{ij}` to `psi_vector`.
    pair_coefficients: ndarray
        Coefficients :math:`c_{ij}` for each :math:`\\tau_{ij}`.

    Returns
    -------
    ndarray
        [spin_isospin] :math:`\\sum_{i<j} c_{ij} \\tau_{ij} |\\Psi(R)\\rangle.

    """
    psi_r = get_psi_r(psi, psi_params, r_coords, psi_vector)
    return tau_psi_r(psi_r, iso_spin_exchange_indices, pair_coefficients)


def sigma(psi, psi_params, psi_vector, r_coords, spin_exchange_indices, pair_coefficients):
    """Evaluate sum_ij sigma_ij on psi_r first evaluate psi_r using vector and psi

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    spin_exchange_indices: ndarray
        2D array containing the indices after applying :math:`\\sigma_{ij}` to `psi_vector`.
    pair_coefficients: ndarray
        Coefficients :math:`c_{ij}` for each :math:`\\simga_{ij}`.

    Returns
    -------
    ndarray
        [spin_isospin] :math:`\\sum_{i<j} c_{ij} \\simga_{ij} |\\Psi(R)\\rangle.

    """
    psi_r = get_psi_r(psi, psi_params, r_coords, psi_vector)
    return sigma_psi_r(psi_r, spin_exchange_indices, pair_coefficients)


def sigma_psi_r(psi_r, exchange_indices, pair_coefficients):
    """Evaluate sum_ij sigma_ij on psi_r

    Parameters
    ----------
    psi_r: ndarray
        The wave function evaluated at R.
    exchange_indices: ndarray
        2D array containing the indices after applying :math:`\\sigma_{ij}` or :math:`\\tau_{ij}`
        denoted :math:`O_{ij}` to the wave function.
    pair_coefficients: ndarray
        Coefficients :math:`c_{ij}` for each exchange operator.

    Returns
    -------
    ndarray
        [spin_isospin] :math:`\\sum_{i<j} c_{ij} \\O_{ij} |\\Psi(R)\\rangle.

    """
    exchanged_psi_r = psi_r[:, exchange_indices]  # [n_isospin, n_spin, n_pair_exchanges]
    psi_r = jnp.expand_dims(psi_r, -1)  # [n_isospin, n_spin, 1]
    psi_r_prime = 2.0 * exchanged_psi_r - psi_r
    psi_r_prime *= pair_coefficients
    psi_r_prime = psi_r_prime.sum(-1)
    return psi_r_prime


def tau_psi_r(psi_r, exchange_indices, pair_coefficients):
    """Evaluate sum_ij tau_ij on psi_r

    Parameters
    ----------
    psi_r: ndarray
        The wave function evaluated at R.
    exchange_indices: ndarray
        2D array containing the indices after applying :math:`\\sigma_{ij}` or :math:`\\tau_{ij}`
        denoted :math:`O_{ij}` to the wave function.
    pair_coefficients: ndarray
        Coefficients :math:`c_{ij}` for each exchange operator.

    Returns
    -------
    ndarray
        [spin_isospin] :math:`\\sum_{i<j} c_{ij} \\O_{ij} |\\Psi(R)\\rangle.

    """
    exchanged_psi_r = psi_r[exchange_indices]  # [n_isospin, n_spin, n_pair_exchanges]
    psi_r = jnp.expand_dims(psi_r, 1)  # [n_isospin, n_spin, 1]
    psi_r_prime = 2.0 * exchanged_psi_r - psi_r
    pair_coefficients = pair_coefficients.reshape(-1, 1)
    psi_r_prime *= pair_coefficients
    psi_r_prime = psi_r_prime.sum(1)
    return psi_r_prime


def get_tau_ij(psi_r, isospin_exchange_indices, pair_coefficients):
    isospin_exchange_indices = jnp.expand_dims(isospin_exchange_indices, -1)
    tau_ij = vmap(tau_psi_r, in_axes=(None, 1, 0))(psi_r, isospin_exchange_indices, pair_coefficients)
    return tau_ij


def get_sigma_ij(psi_r, spin_exchange_indices, pair_coefficients):
    spin_exchange_indices = jnp.expand_dims(spin_exchange_indices, -1)
    sigma_ij = vmap(sigma_psi_r, in_axes=(0, 1, None))(psi_r, spin_exchange_indices, pair_coefficients)
    return sigma_ij


def sigma_tau_psi_r(psi_r, spin_exchange_indices, isospin_exchange_indices, pair_coefficients):
    tau_ij = get_tau_ij(psi_r, isospin_exchange_indices, pair_coefficients)
    sigma_tau_ij = get_sigma_ij(tau_ij, spin_exchange_indices, 1.0)
    sum_sigma_tau_ij = sigma_tau_ij.sum(axis=0)
    return sum_sigma_tau_ij
