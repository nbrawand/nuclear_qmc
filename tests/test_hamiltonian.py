from nuclear_qmc.operators.hamiltonian import get_r_ij_sqrd, get_r_ik_r_ij_cycles, get_local_energy, \
    potential_energy_psi
import jax.numpy as jnp
from nuclear_qmc.operators.operators import kinetic_energy_psi
from jax.config import config
from nuclear_qmc.wave_function.jastro import exponential_jastro as exp_psi

from nuclear_qmc.wave_function.legacy_wave_function_for_testing.test_neural_network import build_test_nn_wfc
from nuclear_qmc.wave_function.utility import get_wave_function_system

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
        psi = exp_psi
        psi_params = jnp.array([1.])
        particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 2,
            dtype=jnp.float64,
            as_jax_array=True)
        v_psi = potential_energy_psi(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
                                     spin_exchange_indices)
        wfc_r = psi(psi_params, r_coords) * psi_vector
        psi_v_psi = jnp.vdot(wfc_r, v_psi)
        computed = psi_v_psi / jnp.vdot(wfc_r, wfc_r)
        computed -= 0.73616035  # remove spin dependent PE term
        expected = jnp.array(
            -20.937521470679798) + 0.59135578  # remove spin dependent term AFDMC should only agree on average
        assert jnp.array_equal(computed.round(7), expected.round(7))

    def test_get_local_energy(self):
        r_coords = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        _, psi, psi_params = build_test_nn_wfc()
        particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 1,
            dtype=jnp.float64,
            as_jax_array=True)
        computed = get_local_energy(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
                                    spin_exchange_indices).round(8)
        expected = jnp.array(-2.42576814)
        assert jnp.array_equal(computed, expected)

    def test_potential_energy_with_test_wfc(self):
        _, psi, psi_params = build_test_nn_wfc()
        particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 1,
            dtype=jnp.float64,
            as_jax_array=True)
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        wfc_r = psi(psi_params, ex_r) * psi_vector
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        v_psi = potential_energy_psi(psi, psi_params, psi_vector, ex_r, particle_pairs, particle_triplets,
                                     spin_exchange_indices)
        psi_v_psi = jnp.vdot(wfc_r, v_psi)
        computed = psi_v_psi / psi_norm
        computed = computed.round(8)
        expected = jnp.array(-241.11734481)
        assert jnp.array_equal(computed, expected)

    def test_kinetic_energy_with_test_wfc(self):
        _, psi, psi_params = build_test_nn_wfc()
        particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 1,
            dtype=jnp.float64,
            as_jax_array=True)
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        expected = 238.69157666
        wfc_r = psi(psi_params, ex_r) * psi_vector
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        ke_psi = kinetic_energy_psi(psi, psi_params, ex_r) * psi_vector
        psi_ke_psi = jnp.vdot(wfc_r, ke_psi)
        computed = psi_ke_psi / psi_norm
        computed = computed.round(8)
        assert jnp.array_equal(computed, jnp.array(expected, dtype=jnp.float64))

    def test_psi_r_test_wfc(self):
        _, psi, psi_params = build_test_nn_wfc()
        ex_r = jnp.array([[0.43, 0, 0], [0, 0, 0]])
        expected = jnp.array(0.05251155, dtype=jnp.float64)
        computed = round(psi(psi_params, ex_r), 8)
        assert jnp.array_equal(expected, computed)

    def test_kinetic_energy_with_3H(self):
        psi, psi_params = exp_psi, jnp.array([1.])
        particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 2,
            dtype=jnp.float64,
            as_jax_array=True)
        #ex_r = jnp.ones(9).reshape(3,3)
        ex_r = jnp.array(
            [
                [-0.36651218, - 0.28230912, 0.72319306],
                [-1.11298546, 0.61603241, 0.8157153],
                [0.26104348, - 0.38107508, 0.07886705],
            ]
        )
        expected = 33.38246175
        wfc_r = psi(psi_params, ex_r) * psi_vector
        psi_norm = jnp.vdot(wfc_r, wfc_r)
        ke_psi = kinetic_energy_psi(psi, psi_params, ex_r) * psi_vector
        psi_ke_psi = jnp.vdot(wfc_r, ke_psi)
        computed = psi_ke_psi / psi_norm
        computed = computed.round(8)
        assert jnp.array_equal(computed, jnp.array(expected, dtype=jnp.float64))
