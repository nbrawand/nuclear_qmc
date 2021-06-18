import jax.numpy as jnp
from jax.ops import index, index_update
from jax import vmap

from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M


def partial_i_j(f, x, i, j):
    epsilon = 1e-3
    delta = jnp.zeros_like(x)
    delta = index_update(delta, index[i, j], epsilon / 2.)
    return (f(x + delta) - f(x - delta)) / epsilon


def partial_partial(f, x, i, j):
    epsilon = 1e-3
    delta = jnp.zeros_like(x)
    delta = index_update(delta, index[i, j], epsilon)
    return (f(x + delta) - 2.0 * f(x) + f(x - delta)) / epsilon ** 2


def laplacian(f, x, i, j):
    l = vmap(partial_partial, in_axes=(None, None, None, 0))(f, x, i, j)
    l = l.sum(axis=0)
    return l


def kinetic_energy(f, x, i, j):
    ke = vmap(laplacian, in_axes=(None, None, 0, None))(f, x, i, j)
    ke = ke.sum(axis=0)
    ke = - H_BAR_SQRD_OVER_2_M * ke
    return ke
