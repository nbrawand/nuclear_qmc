from nuclear_qmc.utils.finite_difference import kinetic_energy
import jax.numpy as jnp


def test_kinetic_energy():
    f = lambda x: x ** 2
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    i = jnp.arange(2)
    j = jnp.arange(2)
    computed = kinetic_energy(f, x, i, j)
