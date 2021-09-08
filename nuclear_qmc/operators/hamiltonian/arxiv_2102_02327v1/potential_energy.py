from nuclear_qmc.operators.iso_tensor import get_iso_tensor_T_ij
from nuclear_qmc.operators.operators import sigma_ij_psi_r, sigma_psi_r, tau_psi_r, sigma_tau_psi_r, tau_ij_psi_r
from operator import mul

from nuclear_qmc.operators.spin_orbit import spin_orbit_operator
from nuclear_qmc.operators.tensor_forces import get_sij_psi_r, get_flip_indices, get_sigma_operator_prefactors, \
    get_sij_tauij_psi_r, get_bit, make_negative_1_if_spin_down_else_1
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_spin_isospin_indices import get_raw_isospin_indices
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


def get_iso_z_factors(mass_number, proton_number):
    _, indices = get_raw_isospin_indices(mass_number, proton_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)
    extracted_bits = vmap(lambda particle: vmap(get_bit, in_axes=(0, None))(indices, particle))(particle_index)
    z_prefactors = make_negative_1_if_spin_down_else_1(extracted_bits)
    return z_prefactors


def build_arxiv_2102_02327v1(particle_pairs
                             , particle_triplets
                             , spin_exchange_indices
                             , isospin_exchange_indices
                             , isospin_binary_representation
                             , model_string='o'
                             , R3=1.5
                             , include_3body=True
                             , theory_order='lo'):
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

    theory_order = theory_order.lower()

    C01 = {
        'lo': {'a': -4.38524414, 'b': -5.72220536, 'c': -7.00250932, 'd': -8.22926713, 'o': -5.27518671},
        'nlo': {'a': -5.11051122, 'b': -5.15205193, 'c': -5.24089036, 'd': -5.34645335, 'o': -5.14608170}
    }[theory_order][model_string]
    C10 = {
        'lo': {'a': -8.00783936, 'b': -9.34392090, 'c': -10.7734100, 'd': -12.2993164, 'o': -7.04040080},
        'nlo': {'a': -4.22732988, 'b': -4.86213195, 'c': -1.47490885, 'd': -4.42765927, 'o': -5.64430900}
    }[theory_order][model_string]
    R0 = {'a': 1.7, 'b': 1.9, 'c': 2.1, 'd': 2.3, 'o': 1.54592984}[model_string]
    R1 = {'a': 1.5, 'b': 2.0, 'c': 2.5, 'd': 3.0, 'o': 1.83039397}[model_string]

    # no v^EM
    cE = {
        'lo': {
            'a': {1.0: 1.8354, 1.5: 4.6301, 2.0: 11.6871, 2.5: 27.4702},
            'b': {1.0: 0.02828, 1.5: 0.06903, 2.0: 0.16387, 2.5: 20.36545},
            'c': {1.0: -2.09231, 1.5: -5.37280, 2.0: -12.4415, 2.5: -26.8473},
            'd': {1.0: -3.89132, 1.5: -10.9436, 2.0: -25.3577, 2.5: -53.7786},
            'o': {1.0: 1.0786, 1.5: 2.7676, 2.0: 6.95356, 2.5: 16.21993}
        },
        'nlo': {
            'a': {1.0: 0.14877, 1.5: 0.38897, 2.0: 0.97039, 2.5: 2.24176},
            'b': {1.0: 0.33198, 1.5: 0.86155, 2.0: 2.14635, 2.5: 4.95746},
            'c': {1.0: -0.47519, 1.5: -1.23710, 2.0: -3.02891, 2.5: -6.87885},
            'd': {1.0: -0.58694, 1.5: -1.46947, 2.0: -3.50072, 2.5: -7.80518},
            'o': {1.0: 0.35211, 1.5: 0.91745, 2.0: 2.29135, 2.5: 5.30139}
        }
    }[theory_order][model_string][R3]

    C_nlo_1 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': -1.12720036, 'b': -1.82744818, 'c': -4.12069851, 'd': -4.83330518, 'o': -.938734989}
    }[theory_order][model_string]
    C_nlo_2 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': 0.909366063, 'b': 1.14092429, 'c': 2.51441807, 'd': 1.43873251, 'o': 0.483260368}
    }[theory_order][model_string]
    C_nlo_3 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': .0477208278, 'b': 0.353463551, 'c': 1.31550606, 'd': 1.45157319, 'o': 0.404430893}
    }[theory_order][model_string]
    C_nlo_4 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': -.475987004, 'b': -.249962307, 'c': -.137446534, 'd': 1.43861202, 'o': -.531440872}
    }[theory_order][model_string]
    C_nlo_5 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': .0494135315, 'b': -.005823185, 'c': 0.688507262, 'd': .0347184150, 'o': -.302484884}
    }[theory_order][model_string]
    C_nlo_6 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': -.846846770, 'b': -1.00082249, 'c': -1.80046641, 'd': -1.25608697, 'o': -.621725001}
    }[theory_order][model_string]
    C_nlo_7 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': -1.55550814, 'b': -1.38788868, 'c': -1.50745124, 'd': -1.53475063, 'o': -1.36793827}
    }[theory_order][model_string]

    CIT0 = {
        'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
        'nlo': {'a': .0190747072, 'b': .0242061782, 'c': .0343911021, 'd': .0488093390, 'o': .0219960910}
    }[theory_order][model_string]

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

    # nlo items
    mass_number = int(particle_pairs.reshape(-1).max() + 1)
    sigma_flipped_indices = get_flip_indices(mass_number)
    _, y_sigma_prefactors, z_sigma_prefactors = get_sigma_operator_prefactors(mass_number)

    # iso
    proton_number = int(isospin_binary_representation[0, :].sum())
    z_tau_prefactors = get_iso_z_factors(mass_number, proton_number)

    def C(r_ij, R):
        out = jnp.exp(-(r_ij / R) ** 2)
        out /= jnp.pi ** (3. / 2.)
        out /= R ** 3
        return out

    def C0(r_ij):
        return C(r_ij, R0)

    def C1(r_ij):
        return C(r_ij, R1)

    def d1Calpha(r_ij, R):
        out = -2.0 * r_ij * C(r_ij, R) / R
        return out

    def d2Calpha(r_ij, R):
        out = 2.0 * C(r_ij, R) / R
        out *= ((2 * (r_ij ** 2) / R) - 1)
        return out

    def P_0(o_ij):
        return (1 - o_ij) / 4.0

    def P_1(o_ij):
        return (3 + o_ij) / 4.0

    def C_total(r_ij, tau_ij):
        out = C(r_ij, R0) * P_0(tau_ij) + C(r_ij, R1) * P_1(tau_ij)
        return out

    def d1C(r_ij, tau_ij):
        out = d1Calpha(r_ij, R0) * P_0(tau_ij) + d1Calpha(r_ij, R1) * P_1(tau_ij)
        return out

    def d2C(r_ij, tau_ij):
        out = d2Calpha(r_ij, R0) * P_0(tau_ij) + d2Calpha(r_ij, R1) * P_1(tau_ij)
        return out

    def add_lo_v_c_r(r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += 3. * (C01 * C1(r_ij) + C10 * C0(r_ij)) / 16.
        pair_coefficients = pair_coefficients.sum() * psi_r
        return pair_coefficients

    def add_lo_v_tau_r(r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += (C01 * C1(r_ij) - 3.0 * C10 * C0(r_ij)) / 16.
        out = tau_psi_r(psi_r, isospin_exchange_indices, pair_coefficients)
        return out

    def add_lo_v_sigma_r(r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += (-3.0 * C01 * C1(r_ij) + C10 * C0(r_ij)) / 16.
        out = sigma_psi_r(psi_r, spin_exchange_indices, pair_coefficients)
        return out

    def add_lo_v_sigma_tau_r(r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += -(C01 * C1(r_ij) + C10 * C0(r_ij)) / 16.
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

    def v_nlo_t(delta_r_ij, r_ij, psi_r, sigma_ij, expected_tau_ij, particle_i, particle_j):
        """per ij"""
        sij = get_sij_psi_r(delta_r_ij, psi_r, particle_i, particle_j
                            , sigma_flipped_indices
                            , y_sigma_prefactors
                            , z_sigma_prefactors
                            , sigma_ij)

        sij *= -C_nlo_5 * (d2C(r_ij, expected_tau_ij) - d1C(r_ij, expected_tau_ij) / r_ij)
        return sij

    def v_nlo_t_tau(delta_r_ij, r_ij, tau_ij, sigma_ij, particle_i, particle_j, expected_tau_ij):
        sij_tauij = get_sij_tauij_psi_r(delta_r_ij, tau_ij, particle_i, particle_j
                                        , sigma_flipped_indices, y_sigma_prefactors, z_sigma_prefactors, sigma_ij)
        sij_tauij *= -C_nlo_6 * (d2C(r_ij, expected_tau_ij) - d1C(r_ij, expected_tau_ij) / r_ij)
        return sij_tauij

    def v_nlo_b(r_coords, r_ij, delta_rij, psi, psi_params, expected_tau_ij):
        Lij = spin_orbit_operator(psi, psi_params, r_coords, delta_rij, sigma_flipped_indices
                                  , y_sigma_prefactors, z_sigma_prefactors, particle_pairs)
        Lij *= (-C_nlo_7 * d1C(r_ij, expected_tau_ij) / r_ij)[:, None, None]
        Lij = Lij.sum(axis=0)
        return Lij

    def v_nlo_T(r_ij, psi_r, tau_ij, expected_tau_ij):
        Tij = get_iso_tensor_T_ij(psi_r, particle_pairs, z_tau_prefactors, tau_ij)
        Tij *= (CIT0 * C_total(r_ij, expected_tau_ij))[:, None, None]
        Tij = Tij.sum(axis=0)
        return Tij

    def potential(psi, psi_params, r_coords):
        r_ij = get_r_ij(r_coords, particle_pairs)
        psi_r = get_psi_r(psi, psi_params, r_coords)
        out = 0.0
        if theory_order == 'nlo':
            sigma_ij = sigma_ij_psi_r(psi_r, spin_exchange_indices, 1.0)
            sigma_ij = jnp.moveaxis(sigma_ij, -1, 0)
            tau_ij_psi_r_value = tau_ij_psi_r(psi_r, isospin_exchange_indices, jnp.ones(shape=len(particle_pairs)))
            tau_ij_psi_r_value = jnp.swapaxes(tau_ij_psi_r_value, 0, 1)
            expected_tau_ij = vmap(lambda a: jnp.vdot(a, psi_r))(tau_ij_psi_r_value) / jnp.vdot(
                psi_r,
                psi_r)
            delta_rij = r_coords[particle_pairs[:, 0]] - r_coords[particle_pairs[:, 1]]
            out += vmap(v_nlo_t, in_axes=(0, 0, None, 0, 0, 0, 0))(delta_rij, r_ij, psi_r
                                                                   , sigma_ij
                                                                   , expected_tau_ij
                                                                   , particle_pairs[:, 0]
                                                                   , particle_pairs[:, 1]).sum(axis=0)
            out += vmap(v_nlo_t_tau)(delta_rij, r_ij, tau_ij_psi_r_value
                                     , sigma_ij
                                     , particle_pairs[:, 0]
                                     , particle_pairs[:, 1]
                                     , expected_tau_ij).sum(axis=0)
            out += v_nlo_b(r_coords, r_ij, delta_rij, psi, psi_params, expected_tau_ij)
            out += v_nlo_T(r_ij, psi_r, tau_ij_psi_r_value, expected_tau_ij)
            nlo_linear_pair_coefficients = -d2C(r_ij, expected_tau_ij) - 2.0 * d1C(r_ij, expected_tau_ij) / r_ij
        else:
            nlo_linear_pair_coefficients = 0.0
        out += add_lo_v_c_r(r_ij, psi_r, C_nlo_1 * nlo_linear_pair_coefficients)
        out += add_lo_v_tau_r(r_ij, psi_r, C_nlo_2 * nlo_linear_pair_coefficients)
        out += add_lo_v_sigma_r(r_ij, psi_r, C_nlo_3 * nlo_linear_pair_coefficients)
        out += add_lo_v_sigma_tau_r(r_ij, psi_r, C_nlo_4 * nlo_linear_pair_coefficients)
        out *= H_BAR
        if particle_triplets.shape[0] > 0 and include_3body:
            out += three_body(r_coords, psi_r)
        out += v_coulomb_proton_proton(r_ij, psi_r)
        return out

    return potential
