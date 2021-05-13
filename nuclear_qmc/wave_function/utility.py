import jax
from jax import numpy as jnp


def apply_confining_potential(r):
    """ Boundary condition imposed on multiple particles
    """
    rcm = jnp.mean(r, axis=0)
    r = r - rcm[None, :]
    return jnp.prod(jax.vmap(sp_boundary, in_axes=(0,))(r))


def sp_boundary(r):
    """ Boundary condition imposed on single particle
    """
    sp_conf = jnp.exp(- 0.1 * jnp.sum(r ** 2))

    return sp_conf