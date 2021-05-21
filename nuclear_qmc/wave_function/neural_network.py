import jax
from copy import deepcopy
from operator import add, mul
from collections import OrderedDict
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
import pickle

from nuclear_qmc.wave_function.combine_wave_functions import combine_wave_functions
from nuclear_qmc.wave_function.jastro import build_sigma_jastro, build_3b_jastro, build_2b_jastro, build_tau_jastro
from nuclear_qmc.wave_function.utility import apply_confining_potential


def load_params(params_file_name):
    with open(params_file_name, 'rb') as fil:
        return pickle.load(fil)


def build_nn_wfc(ndense=4, key=None, params_file=None, in_shape=(1,), n_hidden_layers=2,
                 dtype=jnp.float64):
    if key is None:
        key = random.PRNGKey(0)

    hidden_layers = n_hidden_layers * [Dense(ndense), Tanh, ]
    phi_a_init, phi_a_apply = stax.serial(
        *hidden_layers,
        Dense(1),
    )

    if params_file is not None:
        unflattened_params = load_params(params_file)
    else:
        key, key_input = jax.random.split(key)
        _, unflattened_params = phi_a_init(key_input, in_shape)
        unflattened_params = jax.tree_multimap(lambda params: params.astype(jnp.float64), unflattened_params)

    flat_params, unflatten_params_function = ravel_pytree(unflattened_params)

    def psi_prefactor(flat_params_in, nn_input):
        unflat_params = unflatten_params_function(flat_params_in)
        psi_out = phi_a_apply(unflat_params, nn_input)
        return psi_out.reshape()

    return key, psi_prefactor, flat_params


def build_jastro_nn(
        key
        , spin
        , particle_pairs
        , particle_triplets=None
        , spin_exchange_indices=None
        , isospin_exchange_indices=None
        , n_dense=6
        , n_hidden_layers=2
        , jastro_string='2b+3b+spin'
):
    jastro_string = jastro_string.replace('spin', 'sigma')

    # check for bad jastro type
    for s in jastro_string.split('+'):
        if s not in ['2b', '3b', 'sigma', 'tau']:
            raise RuntimeError(s + ' not in supported jastro types')

    # default psi_vector
    if 'sigma' in jastro_string or 'tau' in jastro_string:
        psi_vector = 1.0
    else:
        psi_vector = spin

    # build all functions and parameters
    functions = OrderedDict()
    params = OrderedDict()
    if '2b' in jastro_string:
        key, nn_2b, b2_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_2b = lambda p, r_ij: jnp.exp(nn_2b(p, r_ij))
        params['2b'] = b2_params
        functions['2b'] = build_2b_jastro(func_2b, particle_pairs)

    if '3b' in jastro_string:
        key, nn_3b, b3_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_3b = lambda p, r_ij: jnp.exp(nn_3b(p, r_ij))
        params['3b'] = b3_params
        functions['3b'] = build_3b_jastro(func_3b, particle_pairs, particle_triplets)

    if 'sigma' in jastro_string and 'tau' in jastro_string:
        # sigma
        key, nn_s, s_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_s = lambda p, r_ij: jnp.tanh(nn_s(p, r_ij))
        jastro_s_func = build_sigma_jastro(func_s, particle_pairs, spin, spin_exchange_indices)

        # tau
        key, nn_t, t_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_t = lambda p, r_ij: jnp.tanh(nn_t(p, r_ij))
        jastro_t_func = build_tau_jastro(func_t, particle_pairs, spin, isospin_exchange_indices)

        # combine
        sigma_tau_func, sigma_tau_params = combine_wave_functions(jastro_s_func, s_params, jastro_t_func, t_params, add)
        functions['spin_sigma_tau'] = lambda p, r: spin + sigma_tau_func(p, r)
        params['spin_sigma_tau'] = sigma_tau_params
    elif 'sigma' in jastro_string:
        key, nn_s, s_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_s = lambda p, r_ij: jnp.tanh(nn_s(p, r_ij))
        jastro_s_func = build_sigma_jastro(func_s, particle_pairs, spin, spin_exchange_indices)
        fun = lambda p, r: spin + jastro_s_func(p, r)
        functions['sigma'] = fun
        params['sigma'] = s_params
    elif 'tau' in jastro_string:
        key, nn_t, t_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
        func_t = lambda p, r_ij: jnp.tanh(nn_t(p, r_ij))
        jastro_t_func = build_tau_jastro(func_t, particle_pairs, spin, isospin_exchange_indices)
        fun = lambda p, r: spin + jastro_t_func(p, r)
        functions['tau'] = fun
        params['tau'] = t_params

    # combine functions using product
    _, psi = functions.popitem(last=False)
    _, psi_parameters = params.popitem(last=False)
    for func, param in zip(functions.values(), params.values()):
        psi, psi_parameters = combine_wave_functions(psi, psi_parameters, func, param, mul)

    confined_psi = lambda p, r: psi(p, r) * apply_confining_potential(r)

    return key, confined_psi, psi_parameters, psi_vector
