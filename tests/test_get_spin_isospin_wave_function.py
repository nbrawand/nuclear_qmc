from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
import numpy as np
import jax.numpy as jnp
from nuclear_qmc.spin.spherical_harmonics import get_spherical_harmonic_system


def iterable_equal(a, b):
    return all([aa == bb for aa, bb in zip(a, b)])


class TestGetWFC:
    def test_get_spin_isospin_wave_function_A2(self):
        expected = jnp.array([[1., 0., 0., 0., ],
                              [-1., 0., 0., 0.]], dtype=jnp.float32)
        computed = get_spin_isospin_wave_function(1, 1, dtype=jnp.float32)
        assert jnp.array_equal(expected, computed)

    def test_get_spin_isospin_wave_function_A3(self):
        expected = jnp.array(
            [[0., 0., -1., 0., 1., 0., 0., 0.],
             [0., 1., 0., 0., -1., 0., 0., 0.],
             [0., -1., 1., 0., 0., 0., 0., 0.]]
            , dtype=jnp.float32)
        computed = get_spin_isospin_wave_function(1, 2, dtype=jnp.float32)
        assert jnp.array_equal(expected, computed)

    def test_get_spin_isospin_wave_function_6Li(self):
        n_protons = 3
        n_neutrons = 3
        computed = get_spin_isospin_wave_function(n_protons, n_neutrons, dtype=jnp.float32)
        expected = np.load('saved.npy')
        assert jnp.array_equal(computed, expected)

    def test_get_spin_isospin_wave_function_6Li_with_orbitals(self):
        n_protons = 3
        n_neutrons = 3
        n_orbitals = None#['Y', 'Z']
        p_orbitals = None#['Y', 'Z']
        computed = get_spin_isospin_wave_function(n_protons, n_neutrons, dtype=jnp.float32, neutron_orbitals=n_orbitals,
                                                  proton_orbitals=p_orbitals)
        expected = np.load('saved.npy')
        assert jnp.array_equal(computed, expected)

    def test_get_spherical_harmonic_system_not_including_functions(self):
        L = 0
        Lz = 0
        L1 = 1
        L2 = 1
        names, coefs, funcs = get_spherical_harmonic_system(L, Lz, L1, L2)

        expected_names = np.array([['Y_1_-1', 'Y_1_1'], ['Y_1_0', 'Y_1_0'], ['Y_1_1', 'Y_1_-1']])
        assert np.array_equal(expected_names, names)

        expected_coefs = jnp.array([np.sqrt(3) / 3., -np.sqrt(3) / 3., np.sqrt(3) / 3.])
        assert jnp.array_equal(expected_coefs, coefs)
