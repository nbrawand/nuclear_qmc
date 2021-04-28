import jax
import jax.numpy as jnp
from jax import random, jit
from jax.ops import index, index_update
from jax.lax import fori_loop
from functools import partial


@partial(jit, static_argnums=(0,))
def sample(key
           , initial_walker_standard_deviation
           , walker_step_size
           , n_walkers
           , n_particles
           , n_dimensions
           , n_equilibrium_steps
           , n_steps
           , n_void_steps
           , wave_function):
    def step_nvoid(i, loop_carry_i):
        key, x_o, acc_x_tot, x_stored, sz_stored = loop_carry_i
        key, *key_input = random.split(key, num=3)
        move = walker_step_size * jax.random.normal(key_input[0], shape=[n_void_steps, n_walkers, n_particles, n_dimensions])
        unif_x = jax.random.uniform(key_input[1], shape=[n_void_steps, n_walkers])

        def step(j, loop_carry_j):
            x_o, wpsi_o, acc_x = loop_carry_j
            x_n = x_o + move[j, :, :, :]
            wpsi_n = wave_function.weight(x_n)
            prob = (jnp.abs(wpsi_n) / jnp.abs(wpsi_o)) ** 2
            accept = jnp.greater_equal(prob, unif_x[j, :])
            x_o = jnp.where(accept.reshape([n_walkers, 1, 1]), x_n, x_o)
            wpsi_o = jnp.where(accept, wpsi_n, wpsi_o)
            acc_x += jnp.mean(accept.astype('float64'))
            return x_o, wpsi_o, acc_x

        xcm = jnp.mean(x_o, axis=1)
        x_o = x_o - xcm[:, None, :]
        wpsi_o = wave_function.weight(x_o)
        acc_x = 0

        x_o, wpsi_o, acc_x = fori_loop(0, n_void_steps, step, (x_o, wpsi_o, acc_x))

        acc_x_tot += acc_x / n_void_steps
        x_stored = index_update(x_stored, index[i, :, :, :], x_o)

        return key, x_o, acc_x_tot, x_stored

    key, key_input = jax.random.split(key)
    x_o = initial_walker_standard_deviation * jax.random.normal(key_input, shape=[n_walkers, n_particles, n_dimensions])
    # Equilibrium steps
    acc_x_tot = 0
    x_stored = jnp.zeros(shape=[n_equilibrium_steps, n_walkers, n_particles, n_dimensions])
    key, x_o, acc_x_tot, x_stored = fori_loop(0, n_equilibrium_steps, step_nvoid, (
        key, x_o, acc_x_tot, x_stored))

    # Averaging steps
    acc_x_tot = 0
    x_stored = jnp.zeros(shape=[n_steps, n_walkers, n_particles, n_dimensions])
    key, x_o, acc_x_tot, x_stored = fori_loop(0, n_steps, step_nvoid, (
        key, x_o, acc_x_tot, x_stored))
    acc_x_tot = acc_x_tot / (n_steps)

    return x_stored, acc_x_tot, key
