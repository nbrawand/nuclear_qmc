from jax.config import config
import jax.numpy as jnp
import json
from nuclear_qmc.optimize.optimize_wave_function import optimize_wave_function
from nuclear_qmc.utils.get_new_file_name import get_new_file_name
from nuclear_qmc.wave_function.neural_network import build_jastro_nn
from nuclear_qmc.wave_function.wave_function import get_wave_function_system
import os
from jax import random
import logging
import argparse

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='This routine performs VMC calculations for nuclear systems.')
parser.add_argument("-i", dest="input_file_name", required=True,
                    help="input file specifying system and VMC parameters", metavar="FILE")
args = parser.parse_args()
input_file = open(args.input_file_name, 'r')
input_json = json.load(input_file)
input_json_directory = os.path.dirname(os.path.realpath(input_file.name))
log_file = os.path.join(input_json_directory, 'nuclear_qmc.md')
logging.basicConfig(filename=log_file
                    , format='%(message)s'
                    , level=logging.INFO)

logging.info('# Nuclear QMC Run')
logging.info('## Log File')
logging.info(log_file)
logging.info('## Input File')
logging.info("```json")
logging.info(json.dumps(input_json, indent=4, sort_keys=True))
logging.info("```")

logging.info('## Building Wave Function System')
particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    input_json['n_proton'], input_json['n_neutron'])
key = random.PRNGKey(input_json['wave_function']['seed'])
key, psi_prefactor, psi_params, psi_vector = build_jastro_nn(
    key
    , spin
    , particle_pairs
    , particle_triplets=particle_triplets
    , spin_exchange_indices=spin_exchange_indices
    , n_dense=input_json['wave_function']['n_dense']
    , n_hidden_layers=input_json['wave_function']['n_hidden_layers']
    , jastro_string=input_json['wave_function']['jastro_string']
)

logging.info('## Wave Function Parameters')
if 'wave_function_file' not in input_json['wave_function']:
    # file not present in input create available file name
    input_json['wave_function']['wave_function_file'] = get_new_file_name('wave_function_parameters_0.npy', 0)
    input_json['wave_function']['wave_function_file'] = os.path.join(input_json_directory,
                                                                     input_json['wave_function']['wave_function_file'])
    logging.info(f'creating wave function parameters file: {input_json["wave_function"]["wave_function_file"]}')
else:
    # file present in input
    input_json['wave_function']['wave_function_file'] = os.path.join(input_json_directory
                                                                     ,
                                                                     input_json['wave_function']['wave_function_file'])
    if os.path.isfile(input_json['wave_function']['wave_function_file']):
        # read parameters from existing file
        logging.info(f'reading wave function parameters from: {input_json["wave_function"]["wave_function_file"]}')
        psi_params = jnp.load(input_json['wave_function']['wave_function_file'])  # over write function psi params
    else:
        # create new file with name from input
        logging.info(f'creating wave function parameters file: {input_json["wave_function"]["wave_function_file"]}')

logging.info('## Optimization')
optimize_wave_function(
    input_json['n_proton']
    , input_json['n_neutron']
    , psi_prefactor
    , psi_params
    , psi_vector
    , particle_pairs
    , particle_triplets
    , spin_exchange_indices
    , input_json['wave_function']['wave_function_file']
    , **input_json['optimization']
)
"""
    , seed=input_json['optimization_seed']
    , n_dimensions=input_json['n_dimensions']
    , psi_param_file=psi_param_file
    , n_blocks=input_json['n_blocks']
    , n_equilibrium_blocks=input_json['n_equilibrium_blocks']
    , n_walkers=input_json['n_walkers']
    , n_void_steps=input_json['n_void_steps']
    , walker_step_size=input_json['walker_step_size']
    , initial_walker_standard_deviation=input_json['initial_walker_standard_deviation']
    , n_optimization_steps=input_json['n_optimization_steps']
    , learning_rate=input_json['learning_rate']
    , epsilon_sr=input_json['epsilon_sr']
    , print_local_energy=input_json['print_local_energy']
"""
