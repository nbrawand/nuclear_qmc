from nuclear_qmc.hamiltonian.hamiltonian import get_r_ij_sqrd, get_r_ik_r_ij_cycles, get_local_energy, potential_energy, \
    C_2, C_1
import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M
from nuclear_qmc.wave_function.wave_function import WaveFunction


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

    def test_potential_energy(self):
        wfc = WaveFunction(1, 1)
        r_coords = jnp.array([
            [0, 0, 0]
            , [0, 0, 0]
        ])
        psi_r = wfc.psi(r_coords)
        expected = jnp.vdot(psi_r, psi_r) * (C_1 + C_2)
        computed = potential_energy(wfc, r_coords)
        assert expected == computed

    # def test_get_local_energy(self):
    #     wfc = WaveFunction(1, 1)
    #     r_coords = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    #     computed = get_local_energy(wfc, r_coords)
    #     print(computed)
