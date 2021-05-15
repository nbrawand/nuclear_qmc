from nuclear_qmc.utils.get_dr_ij import get_r_ij
import jax.numpy as jnp


def test_get_r_ij():
    particle_pairs = jnp.array([
        [0, 1]
        , [0, 2]
        , [1, 2]
    ])
    r_coords = jnp.array([
        [0, 0, 0]
        , [0, 0, 1]
        , [0, 0, 2]
    ])
    expected = jnp.array([1, 2, 1])
    computed = get_r_ij(r_coords, particle_pairs)
    assert jnp.array_equal(expected, computed)
