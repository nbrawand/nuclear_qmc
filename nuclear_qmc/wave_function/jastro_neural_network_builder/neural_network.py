from operator import add, mul
from copy import deepcopy
from collections import OrderedDict
import jax.numpy as jnp

from nuclear_qmc.wave_function.build_angular_momentum_wave_function import build_angular_momentum_wave_function
from nuclear_qmc.wave_function.combine_wave_functions import combine_wave_functions
from nuclear_qmc.wave_function.deepset import get_deep_set
from nuclear_qmc.wave_function.jastro import build_sigma_jastro, build_3b_jastro, build_2b_jastro, build_tau_jastro, \
    build_sigma_tau_jastro
from nuclear_qmc.wave_function.jastro_neural_network_builder.get_nn_jastro_func_and_params import \
    get_nn_jastro_func_and_params
from nuclear_qmc.wave_function.utility import apply_confining_potential
from nuclear_qmc.utils.center_particles import center_particles


def add_parentheses_if_needed(expr):
    if '+' in expr:
        expr = '(' + expr + ')'
    return expr


def build_jastro_nn(
        key
        , orbital_psi
        , orbital_psi_params
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
        , isospin_exchange_indices
        , n_particles
        , n_dense=6
        , n_hidden_layers=2
        , jastro_list=None
        , include_distance_in_2b=False
):
    if jastro_list is None:
        jastro_list = ['2b', '3b']

    if include_distance_in_2b and '2b' not in jastro_list:
        raise RuntimeError('2b must be in jastro list if include_distance_in_2b is True.')

    psi_parameters = orbital_psi_params
    n_orbital_params = len(orbital_psi_params)
    key, deep_set_func, deep_set_params = get_deep_set(key, n_dense, n_hidden_layers, (3,), 6)
    psi_parameters = jnp.concatenate((psi_parameters, deep_set_params))


    def psi_function(in_parameters, in_r_coords):
        in_r_coords = center_particles(in_r_coords)
        psi_out = 0
        start = 0
        end = n_orbital_params
        orbitals_psi = orbital_psi(in_parameters[start:end], in_r_coords)
        psi_out += orbitals_psi
        psi_out *= deep_set_func(in_parameters[end:], in_r_coords)
        psi_out *= apply_confining_potential(in_r_coords)
        return psi_out

    psi_vector = 1.0

    return key, psi_function, psi_parameters, psi_vector
