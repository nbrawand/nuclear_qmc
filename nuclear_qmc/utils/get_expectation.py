import jax.numpy as jnp


def get_expectation(psi, a):
    return jnp.vdot(psi, a) / jnp.vdot(psi, psi)
