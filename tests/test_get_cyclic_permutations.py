from nuclear_qmc.utils.get_cyclic_permutations import get_cyclic_permutations
import jax.numpy as jnp


def test_get_cyclic_permutations():
    expected = jnp.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])
    computed = get_cyclic_permutations(jnp.array([0, 1, 2]))
    assert jnp.array_equal(expected, computed)
