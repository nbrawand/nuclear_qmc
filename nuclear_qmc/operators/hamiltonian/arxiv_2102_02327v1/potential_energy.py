from nuclear_qmc.operators.operators import sigma_psi_r, tau_psi_r, sigma_tau_psi_r, get_sigma_ij, get_tau_ij
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.wave_function.utility import get_psi_r
import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR

from nuclear_qmc.utils.get_dr_ij import get_r_ij


def get_01_and_10_channels(spin, spin_exchange_indices, isospin_exchange_indices):
    one = jnp.array([1.0])

    sigma_ij = get_sigma_ij(spin, spin_exchange_indices, one)
    sigma_ij = get_expectation(spin, sigma_ij)

    tau_ij = get_tau_ij(spin, isospin_exchange_indices, one)
    tau_ij = get_expectation(spin, tau_ij)

    p_0 = lambda x: (1. - x) / 4.
    p_1 = lambda x: (3. + x) / 4.

    spin_0_tau_1_channel = p_0(sigma_ij) * p_1(tau_ij)
    spin_1_tau_0_channel = p_1(sigma_ij) * p_0(tau_ij)

    return spin_0_tau_1_channel, spin_1_tau_0_channel


def build_arxiv_2102_02327v1(spin, particle_pairs, spin_exchange_indices, isospin_exchange_indices, model_string='o'):
    """

    Parameters
    ----------
    particle_pairs
    spin_exchange_indices
    isospin_exchange_indices
    model_string

    Returns
    -------
    function

    """
    C01 = {'a': -4.38524414, 'b': -5.72220536, 'c': -7.00250932, 'd': -8.22926713, 'o': -5.27518671}[model_string]
    C10 = {'a': -8.00783936, 'b': -9.34392090, 'c': -10.7734100, 'd': -12.2993164, 'o': -7.04040080}[model_string]
    R0 = {'a': 1.7, 'b': 1.9, 'c': 2.1, 'd': 2.3, 'o': 1.54592984}[model_string]
    R1 = {'a': 1.5, 'b': 2.0, 'c': 2.5, 'd': 3.0, 'o': 1.83039397}[model_string]

    spin_0_tau_1_channel, spin_1_tau_0_channel = get_01_and_10_channels(spin
                                                                        , spin_exchange_indices
                                                                        , isospin_exchange_indices)
    C01 *= spin_0_tau_1_channel
    C10 *= spin_1_tau_0_channel

    def C(r_ij, R):
        out = jnp.exp(-(r_ij / R) ** 2)
        out /= jnp.pi ** (3. / 2.)
        out /= R ** 3
        return out

    def C0(r_ij):
        return C(r_ij, R0)

    def C1(r_ij):
        return C(r_ij, R1)

    def v_c_r(r_ij, psi_r):
        out = C01 * C1(r_ij) + C10 * C0(r_ij)
        out *= 3. / 16.
        out = out.sum() * psi_r
        return out

    def v_tau_r(r_ij, psi_r):
        pair_coefficients = C01 * C1(r_ij) - 3.0 * C10 * C0(r_ij)
        pair_coefficients *= 1. / 16.
        out = tau_psi_r(psi_r, isospin_exchange_indices, pair_coefficients)
        return out

    def v_sigma_r(r_ij, psi_r):
        pair_coefficients = -3.0 * C01 * C1(r_ij) + C10 * C0(r_ij)
        pair_coefficients *= 1. / 16.
        out = sigma_psi_r(psi_r, spin_exchange_indices, pair_coefficients)
        return out

    def v_sigma_tau_r(r_ij, psi_r):
        pair_coefficients = C01 * C1(r_ij) + C10 * C0(r_ij)
        pair_coefficients *= -1. / 16.
        out = sigma_tau_psi_r(psi_r, spin_exchange_indices, isospin_exchange_indices, pair_coefficients)
        return out

    def potential(psi, psi_params, psi_vector, r_coords):
        r_ij = get_r_ij(r_coords, particle_pairs)
        psi_r = get_psi_r(psi, psi_params, r_coords, psi_vector)
        out = v_c_r(r_ij, psi_r)
        out += v_tau_r(r_ij, psi_r)
        out += v_sigma_r(r_ij, psi_r)
        out += v_sigma_tau_r(r_ij, psi_r)
        out *= H_BAR
        return out

    return potential
