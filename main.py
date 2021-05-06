from jax.config import config
import jax

from nuclear_qmc.operators.operators import kinetic_energy_psi
from nuclear_qmc.optimize.optimize import get_new_wave_function_parameters, partial_full_psi_parameters
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
# from nuclear_qmc.wave_function.wave_function_single_orbital import WaveFunctionSingleOrbital as WaveFunction
# from nuclear_qmc.wave_function.wave_function import WaveFunction as WaveFunction
from nuclear_qmc.wave_function.test_neural_network import NeuralNetworkTestWaveFunction as WaveFunction
# from nuclear_qmc.wave_function.exp_network import ExpWaveFunction as WaveFunction
from nuclear_qmc.sampling.sample import sample, center_walkers
from nuclear_qmc.sampling.weight_functions import v_dot_weight

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 0.3
WALKER_STEP_SIZE = 1.0
N_WALKERS = 8000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 100
N_STEPS = 20
N_VOID_STEPS = 100
N_OPTIMIZATION_STEPS = 2000
LEARNING_RATE = 0.0001
wave_function = WaveFunction()
#wave_function = WaveFunction(jnp.array([2.]))

key = random.PRNGKey(SEED)

key, key_input = jax.random.split(key)
x_o = INITIAL_WALKER_STANDARD_DEVIATION * jax.random.normal(key_input, shape=[N_WALKERS, N_PROTON+N_NEUTRON, N_DIMENSIONS],
                                                            dtype=jnp.float64)
x_o = center_walkers(x_o)

rand_count = 0
for n_opt in range(N_OPTIMIZATION_STEPS):
    key, r_coord_samples = sample(
        wave_function
        , v_dot_weight
        , N_STEPS
        , WALKER_STEP_SIZE
        , N_WALKERS
        , N_NEUTRON + N_PROTON
        , N_DIMENSIONS
        , N_EQUILIBRIUM_STEPS
        , N_VOID_STEPS
        , key
        , x_o
    )
    x_o = r_coord_samples[-1]

    r_coord_samples = r_coord_samples.reshape(-1, N_PROTON + N_NEUTRON, N_DIMENSIONS)
    local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coord_samples)
    print('total_energy', local_energy.mean())#, wave_function.params)
    param_updates = get_new_wave_function_parameters(wave_function
                                                     , r_coord_samples
                                                     , LEARNING_RATE
                                                     ,
                                                     partial_function=partial_full_psi_parameters
                                                     ,
                                                     kinetic_energy_operator=kinetic_energy_psi)
    wave_function.params = param_updates
