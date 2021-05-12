from nuclear_qmc.wave_function.jastro import build_jastro_wave_function_with_spin_correlations
import jax.numpy as jnp
from nuclear_qmc.spin.get_tables import get_spin_particle_pairs


def test_jastro_wfc():
    pairs = jnp.array([[0, 1]])
    spin = jnp.array([[1., 2., 3.], [1., 2., 3.]])
    exchange_indices = jnp.array([0, 2, 1, 1, 2, 0]).reshape(3, 2)
    psi = build_jastro_wave_function_with_spin_correlations(pairs, spin, exchange_indices)
    params = jnp.array([0., 0.])
    r = jnp.array([[0, 0, 0], [1, 0, 0]])
    computed = psi(params, r)
    expected = jnp.array([
        [7., 6., 5.],
        [7., 6., 5.]
    ])
    jnp.array_equal(computed, expected)
