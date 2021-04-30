from nuclear_qmc.wave_function.wave_function import WaveFunction
import numpy as np
import jax.numpy as jnp


class TestWaveFunction:
    WFC = WaveFunction(1, 1, include_isospin=True)

    def test_sigma(self):
        assert True

    def test_tau(self):
        assert True

    def test_kinetic_energy(self):
        r_coords = jnp.array([[0., 1., 2.], [0., 0., 0.]])
        computed = self.WFC.kinetic_energy(r_coords)

    def test_tau_sigma(self):
        wfc = jnp.array([[0., 1., 2., 3], [4., 5, 6, 7], [8, 9, 10, 11]])  # 4 spin states, 3 isospin states
        xi = jnp.array([[0, 2, 1, 3], [3, 1, 2, 0]]).T  # 2 pair exchanges
        pair_coefs = jnp.array([0, 1])
        expected = jnp.array([[6., 1., 2., -3.],
                              [10., 5., 6., 1.],
                              [14., 9., 10., 5.]])
        computed = self.WFC._tau_or_sigma(wfc, xi, pair_coefs)
        assert jnp.array_equal(expected, computed)

    def test_tau_sigma_single_exchange(self):
        wfc = jnp.array([[0., 1., 2., 3], [4., 5, 6, 7], [8, 9, 10, 11]])  # 4 spin states, 3 isospin states
        xi = jnp.array([[0, 2, 1, 3]]).T  # 1 pair exchange
        pair_coefs = 1
        expected = jnp.array([[0., 3., 0., 3.],
                              [4., 7., 4., 7.],
                              [8., 11., 8., 11.]])
        computed = self.WFC._tau_or_sigma(wfc, xi, pair_coefs)
        assert jnp.array_equal(expected, computed)
