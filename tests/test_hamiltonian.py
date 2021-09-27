from nuclear_qmc.operators.hamiltonian.get_local_energy import get_local_energy
from jax import random
from nuclear_qmc.wave_function.wave_function_builder.build_wave_function import build_wave_function
from nuclear_qmc.operators.hamiltonian.build_hamiltonian import build_hamiltonian
import jax.numpy as jnp

from nuclear_qmc.operators.hamiltonian.arxiv_2007_14282v2.potential_energy import build_arxiv_2007_14282v2
from nuclear_qmc.operators.operators import kinetic_energy_psi
from jax.config import config

from nuclear_qmc.utils.get_dr_ij import get_r_ij_sqrd, get_r_ik_r_ij_cycles
from nuclear_qmc.wave_function.jastro import exponential_jastro as exp_psi

from nuclear_qmc.wave_function.legacy_wave_function_for_testing.test_neural_network import build_test_nn_wfc
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_system_arrays import \
    get_system_arrays

config.update("jax_enable_x64", True)


class TestHamiltonian:
    def test_get_r_ij_sqrd(self):
        pairs = jnp.array([
            [0, 1]
            , [1, 0]
            , [1, 2]
        ])
        r_coords = jnp.array([
            [1, 1, 2]
            , [1, 1, 1]
            , [3, 2, 1]
        ])
        expected = jnp.array([1, 1, 5])
        computed = get_r_ij_sqrd(r_coords, pairs)
        assert jnp.array_equal(expected, computed)

    def test_get_r_ik_r_ij_cycles_is_callable(self):
        trips = jnp.array([
            [0, 1, 2]
        ])
        r_coords = jnp.array([
            [0, 0, 1]
            , [0, 0, 0]
            , [0, 0, 0]
        ])
        expected = jnp.array([2, 1, 1])
        computed = get_r_ik_r_ij_cycles(r_coords, trips)
        assert jnp.array_equal(computed, expected)

    def test_get_1st_and_3rd_potential_energy_terms_for_A3(self):
        r_coords = jnp.array(
            [
                [-0.36651218, - 0.28230912, 0.72319306],
                [-1.11298546, 0.61603241, 0.8157153],
                [0.26104348, - 0.38107508, 0.07886705],
            ]
        )
        particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices = get_system_arrays(
            1, 2,
            as_jax_array=True)
        key = random.PRNGKey(0)
        key, psi, psi_params = build_wave_function(key, 1, 2, 1, 1
                                                   , states=[['1_1_d_n', '1_1_d_p', '1_1_u_p']]
                                                   , coefficients=jnp.array([1.])
                                                   , confining_factor=1.
                                                   )
        potential_energy_psi = build_arxiv_2007_14282v2(particle_pairs, particle_triplets,
                                                        spin_exchange_indices)
        v_psi = potential_energy_psi(psi, psi_params, r_coords)
        wfc_r = psi(psi_params, r_coords)
        psi_v_psi = jnp.vdot(wfc_r, v_psi)
        computed = psi_v_psi / jnp.vdot(wfc_r, wfc_r)
        computed -= 0.73616035  # remove spin dependent PE term
        expected = jnp.array(
            -20.937521470679798) + 0.59135578  # remove spin dependent term AFDMC should only agree on average
        assert jnp.array_equal(computed.round(7), expected.round(7))

    def test_get_local_energy(self):
        r_coords = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        _, psi, psi_params = build_test_nn_wfc()
        particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices = get_system_arrays(
            1, 1,
            as_jax_array=True)
        key = random.PRNGKey(0)
        key, orbital_psi, orbital_psi_params = build_wave_function(key, 1, 1, 1, 1
                                                                   , states=[['1_1_d_n', '1_1_d_p']]
                                                                   , coefficients=jnp.array([1.])
                                                                   , confining_factor=1.
                                                                   )
        psi_vector = orbital_psi(orbital_psi_params, r_coords)
        psi_func = lambda _p, _r: psi(_p, _r) * psi_vector
        hamiltonian = build_hamiltonian('arxiv_2007_14282v2', particle_pairs, particle_triplets,
                                        spin_exchange_indices, isospin_exchange_indices, use_finite_diff=False)
        computed = get_local_energy(psi_func, psi_params, r_coords, hamiltonian).round(8)
        expected = jnp.array(-2.41785314)
        assert jnp.array_equal(computed, expected)

    def test_potential_energy_with_test_wfc(self):
        _, psif, psi_params = build_test_nn_wfc()
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        key = random.PRNGKey(0)
        key, orbital_psi, orbital_psi_params = build_wave_function(key, 1, 1, 1, 1
                                                                   , states=[['1_1_d_n', '1_1_d_p']]
                                                                   , coefficients=jnp.array([1.])
                                                                   , confining_factor=1.
                                                                   )
        psi_vector = orbital_psi(orbital_psi_params, ex_r)
        psi = lambda _p, _r: psif(_p, _r) * psi_vector
        particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices = get_system_arrays(
            1, 1,
            as_jax_array=True)
        wfc_r = psi(psi_params, ex_r)
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        potential_energy_psi = build_arxiv_2007_14282v2(particle_pairs, particle_triplets,
                                                        spin_exchange_indices)
        v_psi = potential_energy_psi(psi, psi_params, ex_r)
        psi_v_psi = jnp.vdot(wfc_r, v_psi)
        computed = psi_v_psi / psi_norm
        computed = computed.round(8)
        expected = jnp.array(-241.11734481)
        assert jnp.array_equal(computed, expected)

    def test_kinetic_energy_with_test_wfc(self):
        _, psif, psi_params = build_test_nn_wfc()
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        key = random.PRNGKey(0)
        key, orbital_psi, orbital_psi_params = build_wave_function(key, 1, 1, 1, 1
                                                                   , states=[['1_1_d_n', '1_1_d_p']]
                                                                   , coefficients=jnp.array([1.])
                                                                   , confining_factor=1.
                                                                   )
        psi_vector = orbital_psi(orbital_psi_params, ex_r)
        psi = lambda _p, _r: psif(_p, _r) * psi_vector
        expected = 238.69949166
        wfc_r = psi(psi_params, ex_r)
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        ke_psi = kinetic_energy_psi(psi, psi_params, ex_r)
        psi_ke_psi = jnp.vdot(wfc_r, ke_psi)
        computed = psi_ke_psi / psi_norm
        computed = computed.round(8)
        assert jnp.array_equal(computed, jnp.array(expected, dtype=jnp.float64))

    def test_psi_r_test_wfc(self):
        _, psi, psi_params = build_test_nn_wfc()
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        expected = jnp.array(0.05251155, dtype=jnp.float64)
        computed = round(psi(psi_params, ex_r), 7)
        assert jnp.array_equal(expected, computed)

    def test_kinetic_energy_with_3H(self):
        psix, psi_paramsx = exp_psi, jnp.array([1.])
        ex_r = jnp.array(
            [
                [-0.36651218, - 0.28230912, 0.72319306],
                [-1.11298546, 0.61603241, 0.8157153],
                [0.26104348, - 0.38107508, 0.07886705],
            ]
        )
        key = random.PRNGKey(0)
        key, opsi, psi_params = build_wave_function(key, 1, 2, 1, 1
                                                    , states=[['1_1_d_n', '1_1_d_p', '1_1_u_p']]
                                                    , coefficients=jnp.array([1.])
                                                    , confining_factor=0.
                                                    )
        psi = lambda _p, _r: opsi(psi_params, _r) * psix(psi_paramsx, _r)
        expected = 33.38356871
        wfc_r = psi(psi_params, ex_r)
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        ke_psi = kinetic_energy_psi(psi, psi_params, ex_r)
        psi_ke_psi = jnp.vdot(wfc_r, ke_psi)
        computed = psi_ke_psi / psi_norm
        computed = computed.round(8)
        assert jnp.array_equal(computed, jnp.array(expected, dtype=jnp.float64))
