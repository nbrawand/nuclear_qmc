from nuclear_qmc.operators.iso_tensor import get_iso_tensor_T_ij
from nuclear_qmc.operators.operators import sigma_ij_psi_r, sigma_psi_r, tau_psi_r, sigma_tau_psi_r, tau_ij_psi_r
from nuclear_qmc.operators.hamiltonian.em_interaction import build_v_coulomb_proton_proton, \
    build_darwin_foldy_potential, build_v_two_photon_proton_proton
from operator import mul

from nuclear_qmc.operators.spin_orbit import spin_orbit_operator
from nuclear_qmc.operators.tensor_forces import get_sij_psi_r, get_flip_indices, get_sigma_operator_prefactors, \
    get_sij_tauij_psi_r, get_bit, make_negative_1_if_spin_down_else_1
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_system_arrays import get_raw_isospin_indices
from nuclear_qmc.wave_function.utility import get_psi_r
import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR, ALPHA
from jax import vmap

from nuclear_qmc.utils.get_dr_ij import get_r_ij, get_r_ik_r_ij_cycles


def get_iso_z_factors(mass_number, proton_number):
    _, indices = get_raw_isospin_indices(mass_number, proton_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)
    extracted_bits = vmap(lambda particle: vmap(get_bit, in_axes=(0, None))(indices, particle))(particle_index)
    z_prefactors = make_negative_1_if_spin_down_else_1(extracted_bits)
    return z_prefactors


class Arxiv_2102_02327v1_Potential:

    def __init__(self
                 , particle_pairs
                 , particle_triplets
                 , spin_exchange_indices
                 , isospin_exchange_indices
                 , isospin_binary_representation
                 , model_string='o'
                 , R3=1.5
                 , include_3body=True
                 , theory_order='lo'
                 ):

        self.particle_pairs = particle_pairs
        self.particle_triplets = particle_triplets
        self.spin_exchange_indices = spin_exchange_indices
        self.isospin_exchange_indices = isospin_exchange_indices
        self.isospin_binary_representation = isospin_binary_representation
        self.model_string = model_string
        self.R3 = R3
        self.include_3body = include_3body
        self.theory_order = theory_order

        self.theory_order = theory_order.lower()

        self.C01 = {
            'lo': {'a': -4.38524414, 'b': -5.72220536, 'c': -7.00250932, 'd': -8.22926713, 'o': -5.27518671},
            'nlo': {'a': -5.11051122, 'b': -5.15205193, 'c': -5.24089036, 'd': -5.34645335, 'o': -5.14608170}
        }[theory_order][model_string]
        self.C10 = {
            'lo': {'a': -8.00783936, 'b': -9.34392090, 'c': -10.7734100, 'd': -12.2993164, 'o': -7.04040080},
            'nlo': {'a': -4.22732988, 'b': -4.86213195, 'c': -1.47490885, 'd': -4.42765927, 'o': -5.64430900}
        }[theory_order][model_string]
        self.R0 = {'a': 1.7, 'b': 1.9, 'c': 2.1, 'd': 2.3, 'o': 1.54592984}[model_string]
        self.R1 = {'a': 1.5, 'b': 2.0, 'c': 2.5, 'd': 3.0, 'o': 1.83039397}[model_string]

        # no v^EM
        self.cE = {
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

        self.C_nlo_1 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': -1.12720036, 'b': -1.82744818, 'c': -4.12069851, 'd': -4.83330518, 'o': -.938734989}
        }[theory_order][model_string]
        self.C_nlo_2 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': 0.909366063, 'b': 1.14092429, 'c': 2.51441807, 'd': 1.43873251, 'o': 0.483260368}
        }[theory_order][model_string]
        self.C_nlo_3 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': .0477208278, 'b': 0.353463551, 'c': 1.31550606, 'd': 1.45157319, 'o': 0.404430893}
        }[theory_order][model_string]
        self.C_nlo_4 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': -.475987004, 'b': -.249962307, 'c': -.137446534, 'd': 1.43861202, 'o': -.531440872}
        }[theory_order][model_string]
        self.C_nlo_5 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': .0494135315, 'b': -.005823185, 'c': 0.688507262, 'd': .0347184150, 'o': -.302484884}
        }[theory_order][model_string]
        self.C_nlo_6 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': -.846846770, 'b': -1.00082249, 'c': -1.80046641, 'd': -1.25608697, 'o': -.621725001}
        }[theory_order][model_string]
        self.C_nlo_7 = {
            'lo': {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'o': 0.},
            'nlo': {'a': -1.55550814, 'b': -1.38788868, 'c': -1.50745124, 'd': -1.53475063, 'o': -1.36793827}
        }[theory_order][model_string]

        self.CIT0 = {
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

        self.hbarc6 = H_BAR ** 6
        self.fpi4 = 92.4 ** 4
        self.lambda_chi = 1000.0
        self.pi3 = jnp.pi ** 3
        self.R32 = R3 ** 2
        self.R36 = R3 ** 6
        self.three_body_factor = self.cE * self.hbarc6 / self.fpi4 / self.lambda_chi / self.pi3 / self.R36

        # nlo items
        self.mass_number = int(particle_pairs.reshape(-1).max() + 1)
        self.sigma_flipped_indices = get_flip_indices(self.mass_number)
        _, self.y_sigma_prefactors, self.z_sigma_prefactors = get_sigma_operator_prefactors(self.mass_number)

        # iso
        self.proton_number = int(isospin_binary_representation[0, :].sum())
        self.z_tau_prefactors = get_iso_z_factors(self.mass_number, self.proton_number)

        self.v_coulomb_proton_proton = build_v_coulomb_proton_proton(self.particle_pairs,
                                                                     self.isospin_binary_representation)

    @staticmethod
    def C(r_ij, R):
        out = jnp.exp(-(r_ij / R) ** 2)
        out /= jnp.pi ** (3. / 2.)
        out /= R ** 3
        return out

    def C0(self, r_ij):
        return self.C(r_ij, self.R0)

    def C1(self, r_ij):
        return self.C(r_ij, self.R1)

    def d1Calpha(self, r_ij, R):
        R2 = R ** 2
        out = -2.0 * r_ij * self.C(r_ij, R) / R2
        return out

    def d2Calpha(self, r_ij, R):
        R2 = R ** 2
        rij2 = r_ij ** 2
        out = -2.0 * self.C(r_ij, R) / R2
        out *= (1 - (2 * rij2 / R2))
        return out

    @staticmethod
    def P_0(o_ij):
        return (1 - o_ij) / 4.0

    @staticmethod
    def P_1(o_ij):
        return (3 + o_ij) / 4.0

    def C_total(self, r_ij, tau_ij):
        out = self.C(r_ij, self.R0) * self.P_0(tau_ij) + self.C(r_ij, self.R1) * self.P_1(tau_ij)
        return out

    def d1C(self, r_ij, tau_ij):
        out = self.d1Calpha(r_ij, self.R0) * self.P_0(tau_ij) + self.d1Calpha(r_ij, self.R1) * self.P_1(tau_ij)
        return out

    def d2C(self, r_ij, tau_ij):
        out = self.d2Calpha(r_ij, self.R0) * self.P_0(tau_ij) + self.d2Calpha(r_ij, self.R1) * self.P_1(tau_ij)
        return out

    def add_lo_v_c_r(self, r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += 3. * (self.C01 * self.C1(r_ij) + self.C10 * self.C0(r_ij)) / 16.
        pair_coefficients = pair_coefficients.sum() * psi_r
        return pair_coefficients

    def add_lo_v_tau_r(self, r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += (self.C01 * self.C1(r_ij) - 3.0 * self.C10 * self.C0(r_ij)) / 16.
        out = tau_psi_r(psi_r, self.isospin_exchange_indices, pair_coefficients)
        return out

    def add_lo_v_sigma_r(self, r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += (-3.0 * self.C01 * self.C1(r_ij) + self.C10 * self.C0(r_ij)) / 16.
        out = sigma_psi_r(psi_r, self.spin_exchange_indices, pair_coefficients)
        return out

    def add_lo_v_sigma_tau_r(self, r_ij, psi_r, pair_coefficients=0.0):
        pair_coefficients += -(self.C01 * self.C1(r_ij) + self.C10 * self.C0(r_ij)) / 16.
        out = sigma_tau_psi_r(psi_r, self.spin_exchange_indices, self.isospin_exchange_indices, pair_coefficients)
        return out

    def three_body(self, r_coords, psi_r):
        r_ik_r_ij = get_r_ik_r_ij_cycles(r_coords, self.particle_triplets)
        out = self.three_body_factor * jnp.exp(-r_ik_r_ij / self.R32).sum()
        out *= psi_r
        return out

    def v_nlo_t(self, delta_r_ij, r_ij, psi_r, sigma_ij, expected_tau_ij, particle_i, particle_j):
        """per ij"""
        sij = get_sij_psi_r(delta_r_ij, psi_r, particle_i, particle_j
                            , self.sigma_flipped_indices
                            , self.y_sigma_prefactors
                            , self.z_sigma_prefactors
                            , sigma_ij)

        sij *= -self.C_nlo_5 * (self.d2C(r_ij, expected_tau_ij) - self.d1C(r_ij, expected_tau_ij) / r_ij)
        return sij

    def v_nlo_t_tau(self, delta_r_ij, r_ij, tau_ij, sigma_ij, particle_i, particle_j, expected_tau_ij):
        sij_tauij = get_sij_tauij_psi_r(delta_r_ij, tau_ij, particle_i, particle_j
                                        , self.sigma_flipped_indices, self.y_sigma_prefactors, self.z_sigma_prefactors,
                                        sigma_ij)
        sij_tauij *= -self.C_nlo_6 * (self.d2C(r_ij, expected_tau_ij) - self.d1C(r_ij, expected_tau_ij) / r_ij)
        return sij_tauij

    def v_nlo_b(self, r_coords, r_ij, delta_rij, psi, psi_params, expected_tau_ij):
        Lij = spin_orbit_operator(psi, psi_params, r_coords, delta_rij, self.sigma_flipped_indices
                                  , self.y_sigma_prefactors, self.z_sigma_prefactors, self.particle_pairs)
        Lij *= (-self.C_nlo_7 * self.d1C(r_ij, expected_tau_ij) / r_ij)[:, None, None]
        Lij = Lij.sum(axis=0)
        return Lij

    def v_nlo_T(self, r_ij, psi_r, tau_ij, expected_tau_ij):
        Tij = get_iso_tensor_T_ij(psi_r, self.particle_pairs, self.z_tau_prefactors, tau_ij)
        Tij *= (self.CIT0 * self.C_total(r_ij, expected_tau_ij))[:, None, None]
        Tij = Tij.sum(axis=0)
        return Tij

    def __call__(self, psi, psi_params, r_coords):
        r_ij = get_r_ij(r_coords, self.particle_pairs)
        psi_r = get_psi_r(psi, psi_params, r_coords)
        out = 0.0
        if self.theory_order == 'nlo':
            sigma_ij = sigma_ij_psi_r(psi_r, self.spin_exchange_indices, 1.0)
            sigma_ij = jnp.moveaxis(sigma_ij, -1, 0)
            tau_ij_psi_r_value = tau_ij_psi_r(psi_r, self.isospin_exchange_indices,
                                              jnp.ones(shape=len(self.particle_pairs)))
            tau_ij_psi_r_value = jnp.swapaxes(tau_ij_psi_r_value, 0, 1)
            expected_tau_ij = vmap(lambda tau_psi: get_expectation(psi_r, tau_psi))(tau_ij_psi_r_value)
            delta_rij = r_coords[self.particle_pairs[:, 0]] - r_coords[self.particle_pairs[:, 1]]
            out += vmap(self.v_nlo_t, in_axes=(0, 0, None, 0, 0, 0, 0))(delta_rij, r_ij, psi_r
                                                                        , sigma_ij
                                                                        , expected_tau_ij
                                                                        , self.particle_pairs[:, 0]
                                                                        , self.particle_pairs[:, 1]).sum(axis=0)
            out += vmap(self.v_nlo_t_tau)(delta_rij, r_ij, tau_ij_psi_r_value
                                          , sigma_ij
                                          , self.particle_pairs[:, 0]
                                          , self.particle_pairs[:, 1]
                                          , expected_tau_ij).sum(axis=0)
            out += self.v_nlo_b(r_coords, r_ij, delta_rij, psi, psi_params, expected_tau_ij)
            out += self.v_nlo_T(r_ij, psi_r, tau_ij_psi_r_value, expected_tau_ij)
            nlo_linear_pair_coefficients = -self.d2C(r_ij, expected_tau_ij) - 2.0 * self.d1C(r_ij,
                                                                                             expected_tau_ij) / r_ij
        else:
            nlo_linear_pair_coefficients = 0.0
        out += self.add_lo_v_c_r(r_ij, psi_r, self.C_nlo_1 * nlo_linear_pair_coefficients)
        out += self.add_lo_v_tau_r(r_ij, psi_r, self.C_nlo_2 * nlo_linear_pair_coefficients)
        out += self.add_lo_v_sigma_r(r_ij, psi_r, self.C_nlo_3 * nlo_linear_pair_coefficients)
        out += self.add_lo_v_sigma_tau_r(r_ij, psi_r, self.C_nlo_4 * nlo_linear_pair_coefficients)
        out *= H_BAR
        if self.particle_triplets.shape[0] > 0 and self.include_3body:
            out += self.three_body(r_coords, psi_r)
        out += self.v_coulomb_proton_proton(r_ij, psi_r)
        return out
