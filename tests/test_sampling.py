from jax.config import config
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.wave_function.test_neural_network import NeuralNetworkTestWaveFunction as WaveFunction
from nuclear_qmc.sampling.sample import sample, get_psi_psi_r
from nuclear_qmc.sampling.weight_functions import wave_function_prefactor_weight

config.update("jax_enable_x64", True)


def test_psi_psi_vector():
    def psi(r):
        return jnp.array([1., 0.])

    computed = get_psi_psi_r(psi, None)
    expected = jnp.array([1.])
    jnp.array_equal(computed, expected)


def test_psi_psi_scalar():
    def psi(r):
        return jnp.array([1.])

    computed = get_psi_psi_r(psi, None)
    expected = jnp.array([1.])
    jnp.array_equal(computed, expected)


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

    def psi(r):
        r = r.reshape(-1)
        r2_sum = (r ** 2).sum()
        return jnp.exp(-r2_sum / 2.)

    key, r_coord_samples = sample(
        psi
        , N_STEPS
        , WALKER_STEP_SIZE
        , N_WALKERS
        , N_NEUTRON + N_PROTON
        , N_DIMENSIONS
        , N_EQUILIBRIUM_STEPS
        , N_VOID_STEPS
        , key
        , INITIAL_WALKER_STANDARD_DEVIATION
    )
