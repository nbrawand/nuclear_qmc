from jax.random import PRNGKey

from nuclear_qmc.wave_function.neural_network import build_jastro_nn
from nuclear_qmc.wave_function.utility import get_wave_function_system
import jax.numpy as jnp

particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    1, 2)

def n_non_zero_elms(arr):
    arr = arr[abs(arr) > 10 ** (-8)]
    return len(arr)


def test_build_jastro_nn_sigma_open_spin_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , spin
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_dense=1
        , n_hidden_layers=1
        , jastro_string='2b+sigma'
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = n_non_zero_elms(psi_r)
    assert n_non_zero == 8


def test_build_jastro_nn_sigma_open_tau_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , spin
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_dense=1
        , n_hidden_layers=1
        , jastro_string='2b+tau'
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = n_non_zero_elms(psi_r)
    assert n_non_zero == 8


def test_build_jastro_nn_correct_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , spin
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_dense=1
        , n_hidden_layers=1
        , jastro_string='2b'
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = n_non_zero_elms(psi_r)
    assert n_non_zero == 6
