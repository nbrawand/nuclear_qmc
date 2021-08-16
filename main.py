from jax.config import config
import jax.numpy as jnp
import json

from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from nuclear_qmc.operators.hamiltonian.build_hamiltonian import build_hamiltonian
from nuclear_qmc.optimize.optimize_wave_function import optimize_wave_function
from nuclear_qmc.utils.get_new_file_name import get_new_file_name
from nuclear_qmc.wave_function.neural_network_jastro_builder.add_neural_network_jastros import \
    add_neural_network_jastros
from nuclear_qmc.wave_function.get_spin_isospin_tables.get_spin_isospin_indices import get_spin_isospin_indices
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
# add defaults
if 'potential_energy' not in input_json.keys():
    input_json['potential_energy'] = 'arxiv_2007_14282v2'
if 'potential_kwargs' not in input_json.keys():
    input_json['potential_kwargs'] = None
logging.info("```json")
logging.info(json.dumps(input_json, indent=4, sort_keys=True))
logging.info("```")

logging.info('## Building Wave Function System')
particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices, isospin_binary_representation = get_spin_isospin_indices(
    input_json['n_proton'], input_json['n_neutron'], also_return_binary_representation=True)
key = random.PRNGKey(input_json['wave_function']['seed'])
key, orbital_psi, orbital_psi_params = build_wave_function(key
                                                           , input_json['n_neutron']
                                                           , input_json['n_proton']
                                                           , input_json['wave_function']['n_dense']
                                                           , input_json['wave_function']['n_hidden_layers'])
key, psi_prefactor, psi_params, psi_vector = add_neural_network_jastros(
    key
    , orbital_psi
    , orbital_psi_params
    , particle_pairs
    , particle_triplets=particle_triplets
    , spin_exchange_indices=spin_exchange_indices
    , isospin_exchange_indices=isospin_exchange_indices
    , n_particles=input_json['n_proton'] + input_json['n_neutron']
    , n_dense=input_json['wave_function']['n_dense']
    , n_hidden_layers=input_json['wave_function']['n_hidden_layers']
    , jastro_list=input_json['wave_function']['jastro_list']
)
# logging.info(f'Wave Function Expression: {psi_expression}')

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

logging.info('## Hamiltonian')
hamiltonian = build_hamiltonian(input_json['potential_energy']
                                , particle_pairs
                                , particle_triplets
                                , spin_exchange_indices, isospin_exchange_indices
                                , isospin_binary_representation
                                , input_json['potential_kwargs'])

logging.info('## Optimization')
optimize_wave_function(
    input_json['n_proton']
    , input_json['n_neutron']
    , psi_prefactor
    , psi_params
    , psi_vector
    , input_json['wave_function']['wave_function_file']
    , hamiltonian
    , **input_json['optimization']
)
