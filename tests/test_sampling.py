from jax.config import config
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.sampling.sample import sample, get_psi_psi_r

config.update("jax_enable_x64", True)


def test_psi_psi_vector():
    def psi(p, r):
        return jnp.array([1., 0.])

    one = jnp.array(1.)

    computed = get_psi_psi_r(psi, one, one, one)
    expected = jnp.array([1.])
    jnp.array_equal(computed, expected)


def test_psi_psi_scalar():
    def psi(p, r):
        return jnp.array([1.])

    one = jnp.array(1.)
    computed = get_psi_psi_r(psi, one, one, one)
    expected = jnp.array([1.])
    jnp.array_equal(computed, expected)


"""
def test_sampling():
    N_PROTON = 1
    N_NEUTRON = 0
    SEED = 0
    INITIAL_WALKER_STANDARD_DEVIATION = 1.0
    WALKER_STEP_SIZE = 1.0
    N_WALKERS = 1
    N_DIMENSIONS = 1
    N_EQUILIBRIUM_STEPS = 1
    N_STEPS = 1
    N_VOID_STEPS = 1

    key = random.PRNGKey(SEED)
"""
