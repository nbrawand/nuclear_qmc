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
    """Return a new file name that doesn't exist in current directory.

    Parameters
    ----------
    file_name: str
        Proposed file name. If exists already then append `postfix_int` and check again. Return if doesn't exist.
    postfix_int: int
        Append to end of `file_name` if file already exists.

    Returns
    -------
    file_name: str
        A new file name that doesn't exist.

    """
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
    """Calculate the average local energy.

    Parameters
    ----------
    psi_prefactor: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords_for_block: ndarray
        [n_walkers, n_particles, n_dimensions]
    particle_pairs: ndarray
        [n_pairs, 2] particle indices for each pair.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.
    spin_exchange_indices:
        2D array containing the indices after applying :math:`\\sigma_{ij}` to `psi_vector`.

    Returns
    -------
    local_energy: float
        The local energy.

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
    """
    Optimizes psi_parameters.

    Parameters
    ----------
    n_proton: int
        Number of protons.
    n_neutron: int
        Number of neutrons.
    psi_prefactor: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    particle_pairs: ndarray
        [n_pairs, 2] particle indices for each pair.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.
    spin_exchange_indices:
        2D array containing the indices after applying :math:`\\sigma_{ij}` to `psi_vector`.
    seed: int
        Seed for sampling algorithm.
    n_dimensions: int
        Number of dimensions to sample.
    psi_param_file: str, optional
        File to save `psi_params` to. If None a file will be created.
    n_blocks: int, optional
        Number of blocks to calculate statistics over. Each block has `n_walkers`.
    n_equilibrium_blocks: int, optional
        Number of equilibrium blocks to compute before sampling. Generated arrays are not stored.
    n_walkers: int, optional
        Number of walkers to sample for each block.
    n_void_steps: int, optional
        Number of steps to take before saving a walker during sampling.
    walker_step_size: float, optional
        Standard deviation of gaussian to sample from for each walker at each step.
    initial_walker_standard_deviation: float
        Standard deviation of gaussian to sample initial walker positions during sampling.
    n_optimization_steps: int, optional
       Number of optimization steps to take.
    learning_rate: float, optional
        Size of learning rate for updating `psi_params`.
    epsilon_sr: float, optional
        Size of diagonal for stabilizing the stochastic reconfiguration equations. Reasonable values are between 10^4
        and 10^6.
    print_local_energy: bool, optional
        If True compute and print local energy during optimization loop.

    Returns
    -------
    key, psi_params
        `key` is split during sampling. `psi_params` are updated using the stochastic reconfiguration equations and
        saved to `psi_param_file`.

    """
    # create new wave function parameter file if needed
    if psi_param_file is None:
        psi_param_file = 'wave_function_parameters_0.npy'
        psi_param_file = get_new_param_file_name(psi_param_file, 0)
        logging.info(f'saving wave function parameters to: {psi_param_file}')

    # begin optimization loop
    key = random.PRNGKey(seed)
    n_particles = n_proton + n_neutron
    for n_opt in range(n_optimization_steps):
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
