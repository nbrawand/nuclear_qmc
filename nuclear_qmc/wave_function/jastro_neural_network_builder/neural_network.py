from operator import add, mul
from collections import OrderedDict
import jax.numpy as jnp

from nuclear_qmc.wave_function.build_angular_momentum_wave_function import build_angular_momentum_wave_function
from nuclear_qmc.wave_function.combine_wave_functions import combine_wave_functions
from nuclear_qmc.wave_function.jastro import build_sigma_jastro, build_3b_jastro, build_2b_jastro, build_tau_jastro, \
    build_sigma_tau_jastro
from nuclear_qmc.wave_function.jastro_neural_network_builder.get_nn_jastro_func_and_params import \
    get_nn_jastro_func_and_params
from nuclear_qmc.wave_function.utility import apply_confining_potential


def add_parentheses_if_needed(expr):
    if '+' in expr:
        expr = '(' + expr + ')'
    return expr


def build_jastro_nn(
        key
        , spin
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
        , isospin_exchange_indices
        , n_particles
        , function_permutations
        , iso_indices
        , spin_indices
        , L_total
        , L_z_total
        , L_1
        , L_2
        , n_dense=6
        , n_hidden_layers=2
        , jastro_list=None
):
    if jastro_list is None:
        jastro_list = ['2b', '3b']

    key, orbital_wave_function, orbital_wave_function_params = build_angular_momentum_wave_function(key
                                                                                                    , n_particles
                                                                                                    ,
                                                                                                    function_permutations
                                                                                                    , iso_indices
                                                                                                    , spin_indices
                                                                                                    , L_total
                                                                                                    , L_z_total
                                                                                                    , L_1
                                                                                                    , L_2
                                                                                                    , spin
                                                                                                    , n_dense
                                                                                                    , n_hidden_layers
                                                                                                    )

    builder_dictionary = {
        '2b': {
            'function': get_nn_jastro_func_and_params
            , 'args': [key
                , n_dense
                , n_hidden_layers
                , build_2b_jastro
                , [particle_pairs]
                , jnp.exp]
            , 'correlation_group': 'top'
            , 'combine_function': mul
        }
        , '3b': {
            'function': get_nn_jastro_func_and_params
            , 'args': [key
                , n_dense
                , n_hidden_layers
                , build_3b_jastro
                , [particle_pairs, particle_triplets]
                , jnp.exp]
            , 'correlation_group': 'top'
            , 'combine_function': mul
        }
        , 'sigma_tau': {
            'function': get_nn_jastro_func_and_params
            , 'args': [key
                , n_dense
                , n_hidden_layers
                , build_sigma_tau_jastro
                , [particle_pairs, orbital_wave_function, spin_exchange_indices, isospin_exchange_indices]
                , jnp.tanh]
            , 'correlation_group': 'linear_operators'
            , 'combine_function': add
        }
        , 'sigma': {
            'function': get_nn_jastro_func_and_params
            , 'args': [key
                , n_dense
                , n_hidden_layers
                , build_sigma_jastro
                , [particle_pairs, orbital_wave_function, spin_exchange_indices]
                , jnp.tanh]
            , 'correlation_group': 'linear_operators'
            , 'combine_function': add
        }
        , 'tau': {
            'function': get_nn_jastro_func_and_params
            , 'args': [key
                , n_dense
                , n_hidden_layers
                , build_tau_jastro
                , [particle_pairs, orbital_wave_function, isospin_exchange_indices]
                , jnp.tanh]
            , 'correlation_group': 'linear_operators'
            , 'combine_function': add
        }
    }

    # check for bad jastro type
    for s in jastro_list:
        if s not in builder_dictionary.keys():
            raise RuntimeError(s + ' not in supported jastro types: ', list(builder_dictionary.keys()))

    # build all functions and parameters combine them with final wave function
    func_dict = OrderedDict()
    params_dict = OrderedDict()
    function_expression = OrderedDict()
    for jastro in jastro_list:
        jastro_builder_function = builder_dictionary[jastro]['function']
        jastro_args = builder_dictionary[jastro]['args']
        group = builder_dictionary[jastro]['correlation_group']
        combine_func = builder_dictionary[jastro]['combine_function']
        func_expression = '' if group not in function_expression.keys() else function_expression[group]
        key, func, param = jastro_builder_function(*jastro_args)
        if group in func_dict.keys():
            func_dict[group], params_dict[group], function_expression[group] = combine_wave_functions(func_dict[group]
                                                                                                      ,
                                                                                                      params_dict[group]
                                                                                                      , func_expression
                                                                                                      , func
                                                                                                      , param
                                                                                                      , jastro
                                                                                                      , combine_func)
        else:
            func_dict[group], params_dict[group], function_expression[group] = func, param, jastro

    # if any linear operator terms are present, add plain spin term to the linear group
    lin_group = 'linear_operators'
    if lin_group in func_dict.keys():
        func_dict[lin_group], params_dict[lin_group], function_expression[lin_group] = combine_wave_functions(
            func_dict[lin_group]
            , params_dict[lin_group]
            , function_expression[lin_group]
            , orbital_wave_function
            , orbital_wave_function_params
            , '1'
            , add)
    else:
        # no linear operators so add orbital wave function for product
        func_dict[lin_group] = orbital_wave_function
        params_dict[lin_group] = orbital_wave_function_params
        function_expression[lin_group] = '1'

    # now multiply all groups together
    _, psi = func_dict.popitem(last=False)
    _, psi_parameters = params_dict.popitem(last=False)
    _, psi_expression = function_expression.popitem(last=False)
    psi_expression = add_parentheses_if_needed(psi_expression)
    for func, param, expr in zip(func_dict.values(), params_dict.values(), function_expression.values()):
        expr = add_parentheses_if_needed(expr)
        psi, psi_parameters, psi_expression = combine_wave_functions(psi, psi_parameters, psi_expression
                                                                     , func, param, expr, mul)

    # add a confining potential
    confined_psi = lambda p, r: psi(p, r) * apply_confining_potential(r)

    psi_vector = 1.0

    return key, confined_psi, psi_parameters, psi_vector, psi_expression
