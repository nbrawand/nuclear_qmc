from jax.config import config
from jax import random, vmap
from nuclear_qmc.hamiltonian.hamiltonian import get_local_energy
# from nuclear_qmc.wave_function.wave_function_single_orbital import WaveFunctionSingleOrbital as WaveFunction
# from nuclear_qmc.wave_function.wave_function import WaveFunction as WaveFunction
from nuclear_qmc.wave_function.nn_two_h import TwoBodyNeuralNetwork as WaveFunction
from nuclear_qmc.sample import sample

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 1.0
WALKER_STEP_SIZE = 1.0
N_WALKERS = 2000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 100
N_STEPS = 20
N_VOID_STEPS = 50
# wave_function = WaveFunction(N_PROTON, N_NEUTRON)
key = random.PRNGKey(SEED)
wave_function = WaveFunction(N_DIMENSIONS, N_PROTON + N_NEUTRON, 1, key, 0.0, N_PROTON, N_NEUTRON,
                             params_file='pionless_4_nucleus_2.model')

key, r_coord_samples = sample(
    wave_function
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

print(local_energy.mean())
