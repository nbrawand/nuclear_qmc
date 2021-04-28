from jax.config import config
from jax import random
# from nuclear_qmc.wave_function.wave_function_base import WaveFunctionBase as WaveFunction
from nuclear_qmc.hamiltonian.hamiltonian import kinetic_energy
from nuclear_qmc.wave_function.wave_function_single_orbital import WaveFunctionSingleOrbital as WaveFunction
from nuclear_qmc.sample import sample

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 1.0
WALKER_STEP_SIZE = 1.0
N_WALKERS = 200
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 20
N_STEPS = 10
N_VOID_STEPS = 10
wave_function = WaveFunction(N_PROTON, N_NEUTRON)

key = random.PRNGKey(SEED)
key, r_coords = sample(
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

psi_density_at_r = wave_function.density(r_coords)
print(kinetic_energy(wave_function, r_coords[0, 0], psi_density_at_r[0, 0]))
