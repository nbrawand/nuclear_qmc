from jax.config import config

import jax
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
from nuclear_qmc.optimize.low_memory_optimize import get_delta_params
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.utils.debug.wave_function_plot import get_wave_function_plot
from nuclear_qmc.utils.get_dr_ij import get_r_ij
from nuclear_qmc.wave_function.jastro_neural_network import build_jastro_nn_2_and_3_body
from nuclear_qmc.wave_function.neural_network import build_nn_wfc
from nuclear_qmc.wave_function.utility import apply_confining_potential
from nuclear_qmc.wave_function.wave_function import get_wave_function_system
import os

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 2
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 1.0
WALKER_STEP_SIZE = 0.2
N_WALKERS = 4000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 20
N_STEPS = 1
N_VOID_STEPS = 200
N_OPTIMIZATION_STEPS = 20000
key = random.PRNGKey(SEED)
particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    N_PROTON, N_NEUTRON)
script_dir = os.path.dirname(os.path.realpath(__file__))
n_dense = 8
n_hidden_layers = 2
psi_vector = spin
key, psi_prefactor, psi_params = build_jastro_nn_2_and_3_body(key
                                                              , n_dense
                                                              , particle_pairs
                                                              , particle_triplets, n_hidden_layers=n_hidden_layers)
param_file = os.path.join(script_dir, 'wfc_parameters')
# psi_params = jnp.load(param_file + '.npy')
learning_rate = 0.0001
epsilon_sr = 0.00001
plot_iter = 0

if __name__ == "__main__":
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
        r_coords = r_coord_samples.reshape(-1, N_PROTON + N_NEUTRON, N_DIMENSIONS)
        local_energy = vmap(get_local_energy, in_axes=(None, None, None, 0, None, None, None))(psi_prefactor
                                                                                               , psi_params
                                                                                               , psi_vector
                                                                                               , r_coords
                                                                                               , particle_pairs
                                                                                               , particle_triplets
                                                                                               , spin_exchange_indices)
        delta_params = get_delta_params(
            psi_prefactor
            , psi_params
            , psi_vector
            , r_coords
            , particle_pairs
            , particle_triplets
            , spin_exchange_indices
            , learning_rate
            , eps=epsilon_sr)
        psi_params += delta_params
        print(n_opt, local_energy.mean(), local_energy.std() ** 2)  # , psi_params)  # , wave_function.params)
        jnp.save(param_file, psi_params)

        # plot_iter += 1
        # plt = get_wave_function_plot(psi_prefactor, psi_params, N_PROTON + N_NEUTRON, N_DIMENSIONS, particle_pairs, 4,
        #                              step_size=0.10)
        # plt.savefig(f'wfc_plot_{n_opt}.png')
        # if plot_iter > 6:
        #     plt.clf()
        #     plot_iter = 0
