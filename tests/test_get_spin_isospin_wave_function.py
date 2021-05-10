from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
import jax.numpy as jnp


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
