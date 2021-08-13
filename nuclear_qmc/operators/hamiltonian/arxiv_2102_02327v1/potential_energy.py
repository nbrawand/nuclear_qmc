from nuclear_qmc.operators.operators import sigma_psi_r, tau_psi_r, sigma_tau_psi_r, get_sigma_ij, get_tau_ij
from operator import mul
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.wave_function.utility import get_psi_r
import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR, ALPHA
from jax import vmap

from nuclear_qmc.utils.get_dr_ij import get_r_ij, get_r_ik_r_ij_cycles


def get_proton_proton_projection(particle_pairs, isospin_binary_representation):
    projection = isospin_binary_representation[:, particle_pairs[:, 0], None] * isospin_binary_representation[:,
                                                                                particle_pairs[:, 1], None]
    projection = jnp.swapaxes(projection, 0, 1)  # [pair, isospin state, 1]
    return projection


def build_arxiv_2102_02327v1(particle_pairs
                             , particle_triplets
                             , spin_exchange_indices
                             , isospin_exchange_indices
                             , isospin_binary_representation
                             , model_string='o'
                             , R3=1.5
                             , include_3body=True):
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

    # no v^EM
    cE = {
        'a': {1.0: 1.8354, 1.5: 4.6301, 2.0: 11.6871, 2.5: 27.4702},
        'b': {1.0: 0.02828, 1.5: 0.06903, 2.0: 0.16387, 2.5: 20.36545},
        'c': {1.0: -2.09231, 1.5: -5.37280, 2.0: -12.4415, 2.5: -26.8473},
        'd': {1.0: -3.89132, 1.5: -10.9436, 2.0: -25.3577, 2.5: -53.7786},
        'o': {1.0: 1.0786, 1.5: 2.7676, 2.0: 6.95356, 2.5: 16.21993}
    }[model_string][R3]

    # with v^EM
    # cE = {
    #     'a': {1.0: 1.793374, 1.5: 4.531530, 2.0: 11.44228, 2.5: 26.8957},
    #     'b': {1.0: -0.015077, 1.5: -0.036880, 2.0: -0.087577, 2.5: -0.19526},
    #     'c': {1.0: -2.130138, 1.5: -5.480962, 2.0: -12.69759, 2.5: -27.4026},
    #     'd': {1.0: -3.921656, 1.5: -11.04952, 2.0: -25.61489, 2.5: -54.3297},
    #     'o': {1.0: 1.0362, 1.5: 2.6637, 2.0: 6.69515, 2.5: 15.6184}
    # }[model_string][R3]

    hbarc6 = H_BAR ** 6
    fpi4 = 92.4 ** 4
    lambda_chi = 1000.0
    pi3 = jnp.pi ** 3
    R32 = R3 ** 2
    R36 = R3 ** 6
    three_body_factor = cE * hbarc6 / fpi4 / lambda_chi / pi3 / R36

    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)

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

    def three_body(r_coords, psi_r):
        r_ik_r_ij = get_r_ik_r_ij_cycles(r_coords, particle_triplets)
        out = three_body_factor * jnp.exp(-r_ik_r_ij / R32).sum()
        out *= psi_r
        return out

    def v_coulomb_proton_proton(r_ij, psi_r):
        """https://arxiv.org/pdf/nucl-th/9408016.pdf eqn. 4"""
        b = 4.27
        x = b * r_ij
        x2 = x ** 2
        x3 = x ** 3
        f = 1. - (1 + 11. / 16. * x + 3. / 16. * x2 + 1. / 48. * x3) * jnp.exp(-x)
        v_coulomb_ij = H_BAR * ALPHA * f / r_ij
        r_ij_zero_limit = H_BAR * ALPHA * b * (1. - 11. / 16.)
        v_coulomb_ij = jnp.nan_to_num(v_coulomb_ij, nan=r_ij_zero_limit)
        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = v_coulomb_ij * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out

    def potential(psi, psi_params, psi_vector, r_coords):
        r_ij = get_r_ij(r_coords, particle_pairs)
        psi_r = get_psi_r(psi, psi_params, r_coords, psi_vector)
        out = v_c_r(r_ij, psi_r)
        out += v_tau_r(r_ij, psi_r)
        out += v_sigma_r(r_ij, psi_r)
        out += v_sigma_tau_r(r_ij, psi_r)
        out *= H_BAR
        if particle_triplets.shape[0] > 0 and include_3body:
            out += three_body(r_coords, psi_r)
        out += v_coulomb_proton_proton(r_ij, psi_r)
        return out

    return potential
