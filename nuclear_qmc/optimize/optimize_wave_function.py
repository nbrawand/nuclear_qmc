from nuclear_qmc.operators.hamiltonian import get_local_energy
import logging
from jax import random
from jax.lax import fori_loop
import os
import jax.numpy as jnp
from jax import vmap
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.optimize.low_memory_optimize import get_delta_params


def get_new_param_file_name(file_name, postfix_int):
    if os.path.isfile(file_name):
        new_int = postfix_int + 1
        file_name = file_name.replace(str(postfix_int), str(new_int))
        return get_new_param_file_name(file_name, new_int)
    else:
        return file_name


def get_local_energy_for_block(
        psi_prefactor
        , psi_params
        , psi_vector
        , r_coords_for_block
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
):
    """

    Parameters
    ----------
    psi_prefactor
    psi_params
    psi_vector
    r_coords_for_block: ndarray [n_walkers, n_particles, n_dimensions]
    particle_pairs
    particle_triplets
    spin_exchange_indices

    Returns
    -------

    """
    local_energy_values = vmap(get_local_energy, in_axes=(None, None, None, 0, None, None, None))(psi_prefactor
                                                                                                  , psi_params
                                                                                                  , psi_vector
                                                                                                  , r_coords_for_block
                                                                                                  , particle_pairs
                                                                                                  , particle_triplets
                                                                                                  ,
                                                                                                  spin_exchange_indices)
    local_energy = local_energy_values.mean()
    return local_energy


def optimize_wave_function(
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
        , psi_param_file=None
        , n_blocks=10
        , n_equilibrium_blocks=10
        , n_walkers=4000
        , n_void_steps=200
        , walker_step_size=0.2
        , initial_walker_standard_deviation=1.0
        , n_optimization_steps=500
        , learning_rate=0.0001
        , epsilon_sr=0.0001
        , print_local_energy=True

):
    # create new wave function parameter file if needed
    if psi_param_file is None:
        psi_param_file = 'wave_function_parameters_0.npy'
        psi_param_file = get_new_param_file_name(psi_param_file, 0)
        logging.info(f'saving wave function parameters to: {psi_param_file}')

    # begin optimization loop
    key = random.PRNGKey(seed)
    n_particles = n_proton + n_neutron
    for n_opt in range(n_optimization_steps):
        # sample
        key, r_coord_samples = sample(
            psi_prefactor
            , psi_params
            , psi_vector
            , n_blocks
            , walker_step_size
            , n_walkers
            , n_particles
            , n_dimensions
            , n_equilibrium_blocks
            , n_void_steps
            , key
            , initial_walker_standard_deviation
        )

        # compute and print the local energy
        if print_local_energy:
            local_energy_per_block = vmap(get_local_energy_for_block
                                          , in_axes=(None, None, None, 0, None, None, None))(psi_prefactor
                                                                                             , psi_params
                                                                                             , psi_vector
                                                                                             , r_coord_samples
                                                                                             , particle_pairs
                                                                                             , particle_triplets
                                                                                             , spin_exchange_indices)
            local_energy = local_energy_per_block.mean()
            ddof = 1 if n_blocks > 1 else 0
            local_energy_error = jnp.std(local_energy_per_block, ddof=ddof)
            local_energy_error = jnp.sqrt(local_energy_error)
            logging.info(f'optimization step, local energy, error: {n_opt}, {local_energy}, {local_energy_error}')

        # compute average wave function parameter update over each block
        def sum_delta_params(i, args):
            _delta_params_sum = args[0]
            _params = args[1]
            _delta_params_sum += get_delta_params(
                psi_prefactor
                , _params
                , psi_vector
                , r_coord_samples[i]
                , particle_pairs
                , particle_triplets
                , spin_exchange_indices
                , learning_rate
                , eps=epsilon_sr)
            return _delta_params_sum, _params

        args = (jnp.zeros_like(psi_params), psi_params)
        args = fori_loop(0, n_blocks, sum_delta_params, args)
        delta_params_avg = args[0] / n_blocks
        psi_params += delta_params_avg
        jnp.save(psi_param_file, psi_params)

    return key, psi_params
