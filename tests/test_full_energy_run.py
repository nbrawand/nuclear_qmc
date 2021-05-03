from jax.config import config
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
from nuclear_qmc.wave_function.test_neural_network import NeuralNetworkTestWaveFunction as WaveFunction
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.sampling.weight_functions import wave_function_prefactor_weight

config.update("jax_enable_x64", True)


def test_full_energy_run():
    N_PROTON = 1
    N_NEUTRON = 1
    SEED = 0
    INITIAL_WALKER_STANDARD_DEVIATION = 1.0
    WALKER_STEP_SIZE = 1.0
    N_WALKERS = 2000
    N_DIMENSIONS = 3
    N_EQUILIBRIUM_STEPS = 100
    N_STEPS = 20
    N_VOID_STEPS = 100
    wave_function = WaveFunction()

    key = random.PRNGKey(SEED)

    key, r_coord_samples = sample(
        wave_function
        , wave_function_prefactor_weight
        , N_STEPS
        , INITIAL_WALKER_STANDARD_DEVIATION
        , WALKER_STEP_SIZE
        , N_WALKERS
        , N_NEUTRON + N_PROTON
        , N_DIMENSIONS
        , N_EQUILIBRIUM_STEPS
        , N_VOID_STEPS
        , key
    )

    r_coord_samples = r_coord_samples.reshape(-1, N_PROTON + N_NEUTRON, N_DIMENSIONS)
    local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coord_samples)
    computed = local_energy.mean().round(12)
    expected = jnp.array(-2.20571193860801, dtype=jnp.float64).round(12)
    assert jnp.array_equal(expected, computed)
