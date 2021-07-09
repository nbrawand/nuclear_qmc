from operator import add, mul
from copy import deepcopy
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

    if 'sigma' in jastro_list:
        key, sigma_func, sigma_params = get_nn_jastro_func_and_params(key
                                                                      , n_dense
                                                                      , n_hidden_layers
                                                                      , build_sigma_jastro
                                                                      , [particle_pairs, spin_exchange_indices]
                                                                      , jnp.tanh)
        psi_parameters = jnp.concatenate((psi_parameters, sigma_params))
        n_sigma = len(sigma_params)

    if 'tau' in jastro_list:
        key, tau_func, tau_params = get_nn_jastro_func_and_params(key
                                                                  , n_dense
                                                                  , n_hidden_layers
                                                                  , build_tau_jastro
                                                                  , [particle_pairs, isospin_exchange_indices]
                                                                  , jnp.tanh)
        psi_parameters = jnp.concatenate((psi_parameters, tau_params))
        n_tau = len(tau_params)

    if 'sigma_tau' in jastro_list:
        key, sigtau_func, sigtau_params = get_nn_jastro_func_and_params(key
                                                                        , n_dense
                                                                        , n_hidden_layers
                                                                        , build_sigma_tau_jastro
                                                                        , [particle_pairs, spin_exchange_indices,
                                                                           isospin_exchange_indices]
                                                                        , jnp.tanh)
        psi_parameters = jnp.concatenate((psi_parameters, sigtau_params))
        n_sigma_tau = len(sigtau_params)

    if '2b' in jastro_list:
        in_shape = (2,) if include_distance_in_2b else (1,)
        key, b2_func, b2_params = get_nn_jastro_func_and_params(key
                                                                , n_dense
                                                                , n_hidden_layers
                                                                , build_2b_jastro
                                                                , [particle_pairs, include_distance_in_2b]
                                                                , jnp.exp
                                                                , in_shape=in_shape)
        psi_parameters = jnp.concatenate((psi_parameters, b2_params))
        n_2b = len(b2_params)

    if '3b' in jastro_list:
        if n_particles > 2:
            key, b3_func, b3_params = get_nn_jastro_func_and_params(key
                                                                    , n_dense
                                                                    , n_hidden_layers
                                                                    , build_3b_jastro
                                                                    , [particle_pairs, particle_triplets]
                                                                    , jnp.exp)
            psi_parameters = jnp.concatenate((psi_parameters, b3_params))
            n_3b = len(b3_params)
        else:
            raise RuntimeError('3b jastro requires A>2')

    def psi_function(in_parameters, in_r_coords):
        psi_out = 0

        # apply orbitals
        start = 0
        end = n_orbital_params
        orbitals_psi = orbital_psi(in_parameters[start:end], in_r_coords)
        psi_out += orbitals_psi

        # linear operators act on orbitals and are added to the original wave function
        if 'sigma' in jastro_list:
            start = end
            end += n_sigma
            psi_out += sigma_func(in_parameters[start:end], in_r_coords, orbitals_psi)

        if 'tau' in jastro_list:
            start = end
            end += n_tau
            psi_out += tau_func(in_parameters[start:end], in_r_coords, orbitals_psi)

        if 'sigtau' in jastro_list:
            start = end
            end += n_sigma_tau
            psi_out += sigtau_func(in_parameters[start:end], in_r_coords, orbitals_psi)

        # multiply by 2b and 3b jastros
        if '2b' in jastro_list:
            start = end
            end += n_2b
            psi_out *= b2_func(in_parameters[start:end], in_r_coords)

        if '3b' in jastro_list:
            start = end
            end += n_3b
            psi_out *= b3_func(in_parameters[start:end], in_r_coords)

        psi_out *= apply_confining_potential(in_r_coords)

        return psi_out

    psi_vector = 1.0

    return key, psi_function, psi_parameters, psi_vector
