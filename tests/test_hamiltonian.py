from nuclear_qmc.hamiltonian.hamiltonian import get_r_ij_sqrd
import jax.numpy as jnp


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
