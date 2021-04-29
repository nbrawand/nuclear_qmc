from nuclear_qmc.utils.get_triplets import get_triplets
import jax.numpy as jnp


def test_get_triplets():
    arr = jnp.array([0, 1, 2, 3, 4])
    expected = jnp.array([
        [0, 1, 2]
        , [0, 1, 3]
        , [0, 2, 3]
        , [1, 2, 3]
        , [0, 1, 4]
        , [0, 2, 4]
        , [1, 2, 4]
        , [1, 3, 4]
        , [2, 3, 4]
    ])
    computed = get_triplets(arr)
    jnp.array_equal(computed, expected)
