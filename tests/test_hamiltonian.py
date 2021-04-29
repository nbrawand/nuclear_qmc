from nuclear_qmc.hamiltonian.hamiltonian import get_r_ij_sqrd, get_r_ik_r_ij_cycles
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

    def test_get_r_ik_r_ij_cycles_is_callable(self):
        trips = jnp.array([
            [0, 1, 2]
            , [0, 2, 1]
        ])
        r_coords = jnp.array([
            [1, 1, 2]
            , [1, 1, 1]
            , [3, 2, 1]
        ])
        computed = get_r_ik_r_ij_cycles(r_coords, trips)
