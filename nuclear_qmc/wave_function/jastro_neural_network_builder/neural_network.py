from operator import add, mul
from copy import deepcopy
from collections import OrderedDict
import jax.numpy as jnp
from jax import vmap

from nuclear_qmc.operators.operators import sigma_psi_r, tau_psi_r, sigma_tau_psi_r
from nuclear_qmc.wave_function.build_angular_momentum_wave_function import build_angular_momentum_wave_function
from nuclear_qmc.wave_function.combine_wave_functions import combine_wave_functions
from nuclear_qmc.wave_function.deepset import get_deep_set
from nuclear_qmc.wave_function.jastro import build_sigma_jastro, build_3b_jastro, build_2b_jastro, build_tau_jastro, \
    build_sigma_tau_jastro, build_2b_addition_jastro, build_3b_addition_jastro
from nuclear_qmc.wave_function.jastro_neural_network_builder.get_nn_jastro_func_and_params import \
    get_nn_jastro_func_and_params
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
        jastro_list = []

    if include_distance_in_2b and '2b' not in jastro_list:
        raise RuntimeError('2b must be in jastro list if include_distance_in_2b is True.')

    supported_jastros = ['2b', '3b', 'sigma', 'tau', 'sigma_tau', 'add_2b', 'add_3b', 'deepset', 'total_deepset']
    for jastro in jastro_list:
        if jastro not in supported_jastros:
            raise RuntimeError(f'jastro: {jastro} not supported.')

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

    if 'add_2b' in jastro_list:
        key, add_b2_func, add_b2_params = get_nn_jastro_func_and_params(key
                                                                        , n_dense
                                                                        , n_hidden_layers
                                                                        , build_2b_addition_jastro
                                                                        , [particle_pairs]
                                                                        , jnp.exp)
        psi_parameters = jnp.concatenate((psi_parameters, add_b2_params))
        add_n_2b = len(add_b2_params)

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

    if 'add_3b' in jastro_list:
        if n_particles > 2:
            key, add_b3_func, add_b3_params = get_nn_jastro_func_and_params(key
                                                                            , n_dense
                                                                            , n_hidden_layers
                                                                            , build_3b_addition_jastro
                                                                            , [particle_pairs, particle_triplets]
                                                                            , jnp.exp)
            psi_parameters = jnp.concatenate((psi_parameters, add_b3_params))
            add_n_3b = len(add_b3_params)
        else:
            raise RuntimeError('3b addition jastro requires A>2')

    if 'deepset' in jastro_list:
        key, deepset_func, deepset_params = get_deep_set(key
                                                         , n_dense
                                                         , n_hidden_layers
                                                         , out_shape=1
                                                         , in_shape=(3,)
                                                         , latent_shape=6
                                                         , wrapper_func=jnp.exp)
        n_deepset_params = len(deepset_params)
        psi_parameters = jnp.concatenate((psi_parameters, deepset_params))

    if 'total_deepset' in jastro_list:
        n_pairs = len(particle_pairs)
        key, total_deepset_nn_func, total_deepset_params = get_deep_set(key
                                                                        , n_dense
                                                                        , n_hidden_layers
                                                                        , out_shape=3 * n_pairs + 1
                                                                        , in_shape=(3,)
                                                                        , latent_shape=6
                                                                        , wrapper_func=jnp.exp)

        def total_deepset_func(_p, _r, _psi):
            x = total_deepset_nn_func(_p, _r)
            x = jnp.tanh(jnp.log(x[:3 * n_pairs]))
            sum_ij_sigma = sigma_psi_r(_psi, spin_exchange_indices, x[:n_pairs])
            sum_ij_tau = tau_psi_r(_psi, isospin_exchange_indices, x[n_pairs:2 * n_pairs])
            sum_ij_sigma_tau = sigma_tau_psi_r(_psi, spin_exchange_indices, isospin_exchange_indices,
                                               x[2 * n_pairs:3 * n_pairs])
            return sum_ij_sigma + sum_ij_tau + sum_ij_sigma_tau + _psi

        n_total_deepset_params = len(total_deepset_params)
        psi_parameters = jnp.concatenate((psi_parameters, total_deepset_params))

    def psi_function(in_parameters, in_r_coords):
        in_r_coords = center_particles(in_r_coords)

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

        if 'sigma_tau' in jastro_list:
            start = end
            end += n_sigma_tau
            psi_out += sigtau_func(in_parameters[start:end], in_r_coords, orbitals_psi)

        # multiply by 2b and 3b jastros
        if '2b' in jastro_list:
            start = end
            end += n_2b
            psi_out *= b2_func(in_parameters[start:end], in_r_coords)

        if 'add_2b' in jastro_list:
            start = end
            end += add_n_2b
            psi_out *= add_b2_func(in_parameters[start:end], in_r_coords)

        if '3b' in jastro_list:
            start = end
            end += n_3b
            psi_out *= b3_func(in_parameters[start:end], in_r_coords)

        if 'add_3b' in jastro_list:
            start = end
            end += add_n_3b
            psi_out *= add_b3_func(in_parameters[start:end], in_r_coords)

        if 'deepset' in jastro_list:
            start = end
            end += n_deepset_params
            psi_out *= deepset_func(in_parameters[start:end], in_r_coords)

        if 'total_deepset' in jastro_list:
            start = end
            end += n_total_deepset_params
            psi_out = total_deepset_func(in_parameters[start:end], in_r_coords, psi_out)

        return psi_out

    psi_vector = 1.0

    return key, psi_function, psi_parameters, psi_vector
