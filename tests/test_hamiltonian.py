from nuclear_qmc.hamiltonian.hamiltonian import get_r_ij_sqrd, get_r_ik_r_ij_cycles, one_particle_kinetic_energy, \
    kinetic_energy, get_local_energy
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
            , [0, 2, 1]
        ])
        r_coords = jnp.array([
            [1, 1, 2]
            , [1, 1, 1]
            , [3, 2, 1]
        ])
        computed = get_r_ik_r_ij_cycles(r_coords, trips)

    def test_one_particle_kinetic_energy(self):
        psi = lambda x: (x[0] ** 2 + x[1] ** 2) * jnp.array([1, 0, 0, 0])
        r_coords = jnp.array([0., 1., 2.])
        computed = one_particle_kinetic_energy(psi, r_coords)
        expected = -H_BAR_SQRD_OVER_2_M*4.
        assert jnp.array_equal(computed, expected)

    def test_one_particle_kinetic_energy(self):
        psi = lambda x: (x[0] ** 2 + x[1] ** 2) * jnp.array([1, 0, 0, 0])
        r_coords = jnp.array([[0., 1., 2.], [0., 1., 2.]])
        expected = -H_BAR_SQRD_OVER_2_M*4.*2.
        computed = kinetic_energy(psi, r_coords)
        assert jnp.array_equal(computed, expected)

    def test_get_local_energy(self):
        wfc = WaveFunction(1, 1)
        wfc.spin = jnp.real(wfc.spin)
        r_coords = jnp.array([[0., 1., 2.], [0., 1., 2.]])
        computed = get_local_energy(wfc, r_coords)
        print(computed)
