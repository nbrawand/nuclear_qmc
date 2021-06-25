from jax.random import PRNGKey

from nuclear_qmc.wave_function.jastro_neural_network_builder.neural_network import build_jastro_nn
from nuclear_qmc.wave_function.utility import get_wave_function_system
import jax.numpy as jnp
from nuclear_qmc.wave_function.build_wave_function import build_wave_function

particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
    1, 2)

key = PRNGKey(0)
key, orbital_psi, orbital_psi_params = build_wave_function(key, 1, 2, 1, 1)


def test_build_jastro_nn_sigma_open_spin_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , orbital_psi
        , orbital_psi_params
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_particles=3
        , n_dense=1
        , n_hidden_layers=1
        , jastro_list=['2b', 'sigma']
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = jnp.count_nonzero(psi_r)
    assert n_non_zero == 8


def test_build_jastro_nn_sigma_open_tau_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , orbital_psi
        , orbital_psi_params
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_particles=3
        , n_dense=1
        , n_hidden_layers=1
        , jastro_list=['2b', 'tau']
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = jnp.count_nonzero(psi_r)
    assert n_non_zero == 8


def test_build_jastro_nn_correct_channels():
    key = PRNGKey(0)
    key, psi, psi_parameters, psi_vector = build_jastro_nn(
        key
        , orbital_psi
        , orbital_psi_params
        , particle_pairs
        , particle_triplets=particle_triplets
        , spin_exchange_indices=spin_exchange_indices
        , isospin_exchange_indices=isospin_exchange_indices
        , n_particles=3
        , n_dense=1
        , n_hidden_layers=1
        , jastro_list=['2b']
    )
    r = jnp.arange(9).reshape(3, 3)
    psi_r = psi(psi_parameters, r) * psi_vector
    psi_r = psi_r.reshape(-1)
    n_non_zero = jnp.count_nonzero(psi_r)
    assert n_non_zero == 6
