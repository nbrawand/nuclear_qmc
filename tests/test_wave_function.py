from nuclear_qmc.wave_function.build_angular_momentum_wave_function import build_angular_momentum_wave_function
from nuclear_qmc.wave_function.build_spin_isospin_wave_function import build_spin_isospin
from nuclear_qmc.wave_function.build_wave_function import build_wave_function
import jax.numpy as jnp


class TestWaveFunction:
    def test_build_wave_function(self):
        func = lambda a, b, c, d: (a, b, c, d)
        build_func = lambda arg, kwarg: func
        build_functions = 2 * [build_func]
        args = []
        kwargs = {}
        wfc = build_wave_function(build_functions, args, kwargs)
        print(wfc(None, None))

    def test_build_spin_isospin(self):
        iso_indices = jnp.array([0, 1])
        spin_indices = jnp.array([0, 0])
        perms = jnp.array([[0, 1], [1, 0]])
        wfc = build_spin_isospin(n_particles=2, n_protons=1, iso_indices=iso_indices, spin_indices=spin_indices
                                 , permutations=perms)
        computed = wfc(None, None)[2]
        expected = jnp.array([[1., 0., 0., 0.],
                              [-1., 0., 0., 0.]], dtype=jnp.float64)
        assert jnp.array_equal(computed, expected)

    def test_build_angular_momentum_wave_function(self):
        wfc = build_angular_momentum_wave_function(n_particles=2
                                                   , spherical_harmonic_function_names=['Y_0_0']
                                                   , function_permutations=jnp.array([[0, 1], [1, 0]])
                                                   , iso_indices=jnp.array([0, 1])
                                                   , spin_indices=jnp.array([0, 0]))
        spin_isospin = jnp.array([[1., 0., 0., 0.],
                                  [-1., 0., 0., 0.]], dtype=jnp.float64)
        r = jnp.array([[0, 0, 0], [1, 0, 0]])
        computed = wfc(None, r, spin_isospin)[2]
        expected = jnp.sqrt(1.0 / jnp.pi)
        expected /= 2.0
        expected = expected ** 2 * spin_isospin
        assert jnp.array_equal(computed, expected)  # TODO: there is a rounding error here
