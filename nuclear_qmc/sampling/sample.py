import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.lax import fori_loop
from nuclear_qmc.utils.center_particles import center_walkers


def get_psi_psi_r(psi, psi_params, r_coords):
    """

    Parameters
    ----------
    psi: function
    r_coords: ndarray [n_particles, n_coordinates]

    Returns
    -------

    """
    psi_r = psi(psi_params, r_coords)
    return jnp.vdot(psi_r, psi_r)


def sample(
        psi
        , psi_params
        , n_steps
        , walker_step_size
        , n_walkers
        , n_particles
        , n_dimensions
        , n_equilibrium_steps
        , n_void_steps
        , key
        , initial_walker_standard_deviation
):
    def step_nvoid(i, loop_carry_i):
        key, x_o, x_stored = loop_carry_i
        key, *key_input = random.split(key, num=3)
        move = walker_step_size * jax.random.normal(key_input[0],
                                                    shape=[n_void_steps, n_walkers, n_particles, n_dimensions],
                                                    dtype=jnp.float64)
        unif_x = jax.random.uniform(key_input[1], shape=[n_void_steps, n_walkers], dtype=jnp.float64)

        def step(j, loop_carry_j):
            x_o, wpsi_o, = loop_carry_j
            x_n = x_o + move[j, :, :, :]
            wpsi_n = vmap(get_psi_psi_r, in_axes=(None, None, 0))(psi, psi_params, x_n)
            prob = jnp.abs(wpsi_n) / jnp.abs(wpsi_o)
            accept = jnp.greater_equal(prob, unif_x[j, :])
            x_o = jnp.where(accept.reshape([n_walkers, 1, 1]), x_n, x_o)
            wpsi_o = jnp.where(accept, wpsi_n, wpsi_o)
            return x_o, wpsi_o

        x_o = center_walkers(x_o)
        wpsi_o = vmap(get_psi_psi_r, in_axes=(None, None, 0))(psi, psi_params, x_o)

        x_o, wpsi_o = fori_loop(0, n_void_steps, step, (x_o, wpsi_o))
        x_o = center_walkers(x_o)

        x_stored = x_stored.at[i, :, :, :].set(x_o)

        return key, x_o, x_stored

    key, key_input = jax.random.split(key)
    x_o = initial_walker_standard_deviation * jax.random.normal(key_input,
                                                                shape=[n_walkers, n_particles, n_dimensions],
                                                                dtype=jnp.float64)
    x_o = center_walkers(x_o)

    # Equilibrium steps
    x_stored = jnp.zeros(shape=[n_equilibrium_steps, n_walkers, n_particles, n_dimensions], dtype=jnp.float64)
    key, x_o, x_stored = fori_loop(0, n_equilibrium_steps, step_nvoid, (
        key, x_o, x_stored))

    # Averaging steps
    x_stored = jnp.zeros(shape=[n_steps, n_walkers, n_particles, n_dimensions], dtype=jnp.float64)
    key, x_o, x_stored = fori_loop(0, n_steps, step_nvoid, (
        key, x_o, x_stored))

    return key, x_stored
