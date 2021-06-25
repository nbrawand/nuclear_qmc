from nuclear_qmc.spin.get_spin_isospin_wave_function import build_wave_function
from jax import random
import jax.numpy as jnp


class TestGetWFC:

    def test_build_wave_function_2H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=1
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2)
        r = jnp.ones(shape=(2, 3))
        computed = psi(params, r)
        expected = jnp.array([[1., 0., 0., 0.],
                              [-1., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)

    def test_build_wave_function_3H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=2
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2)
        r = jnp.ones(shape=(3, 3))
        computed = psi(params, r)
        expected = jnp.array([[0., 0., 1., 0., -1., 0., 0., 0.],
                              [0., -1., 0., 0., 1., 0., 0., 0.],
                              [0., 1., -1., 0., 0., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)
