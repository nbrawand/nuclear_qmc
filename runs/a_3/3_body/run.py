from jax.config import config
import jax.numpy as jnp
from nuclear_qmc.optimize.optimize_wave_function import optimize_wave_function
from nuclear_qmc.wave_function.jastro_neural_network import build_jastro_nn_2_and_3_body
from nuclear_qmc.wave_function.wave_function import get_wave_function_system
import os
from jax import random
import logging

logging.basicConfig(filename='nuclear_qmc.log', level=logging.DEBUG)

config.update("jax_enable_x64", True)
n_proton = 1
n_neutron = 2
particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    n_proton
    , n_neutron)
key = random.PRNGKey(0)
n_dense = 8
script_dir = os.path.dirname(os.path.realpath(__file__))
psi_param_file = os.path.join(script_dir, 'wave_function_parameters_0.npy')
n_hidden_layers = 2
key, psi_prefactor, _ = build_jastro_nn_2_and_3_body(key
                                                              , n_dense
                                                              , particle_pairs
                                                              , particle_triplets, n_hidden_layers=n_hidden_layers)
psi_params = jnp.load(psi_param_file)
_, params = optimize_wave_function(
    n_proton
    , n_neutron
    , psi_prefactor
    , psi_params
    , psi_vector
    , particle_pairs
    , particle_triplets
    , spin_exchange_indices
    , seed=0
    , n_dimensions=3
    , psi_param_file=psi_param_file
    , n_blocks=20
    , n_equilibrium_blocks=10
    , n_walkers=4000
    , n_void_steps=200
    , walker_step_size=0.2
    , initial_walker_standard_deviation=1.0
    , n_optimization_steps=400
    , learning_rate=0.0001
    , epsilon_sr=0.0001
    , print_local_energy=True

)
