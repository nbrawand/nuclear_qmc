from jax.config import config
import jax
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
from nuclear_qmc.optimize.optimize import get_new_wave_function_parameters
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.wave_function.test_neural_network import build_test_nn_wfc
from nuclear_qmc.wave_function.wave_function import get_wave_function_system

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 1.0
WALKER_STEP_SIZE = 0.2
N_WALKERS = 4000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 20
N_STEPS = 5
N_VOID_STEPS = 200
N_OPTIMIZATION_STEPS = 20000
key = random.PRNGKey(SEED)
_, psi_prefactor, psi_params = build_test_nn_wfc(params_file=None)
particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    N_PROTON, N_NEUTRON)
learning_rate = 0.0001

for n_opt in range(N_OPTIMIZATION_STEPS):
    key, r_coord_samples = sample(
        psi_prefactor
        , psi_params
        , psi_vector
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
    r_coords = r_coord_samples.reshape(-1, 2, 3)
    local_energy = vmap(get_local_energy, in_axes=(None, None, None, 0, None, None, None))(psi_prefactor
                                                                                           , psi_params
                                                                                           , psi_vector
                                                                                           , r_coords
                                                                                           , particle_pairs
                                                                                           , particle_triplets
                                                                                           , spin_exchange_indices)
    delta_params = get_new_wave_function_parameters(
        psi_prefactor
        , psi_params
        , psi_vector
        , r_coords
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
        , learning_rate)
    psi_params += delta_params
    print(n_opt, local_energy.mean(), local_energy.std() ** 2)  # , wave_function.params)
    jnp.save('wfc.model', psi_params)
