from nuclear_qmc.operators.hamiltonian.get_local_energy import get_local_energy
import logging
from jax import random
from jax.lax import fori_loop
import jax.numpy as jnp
from jax import vmap
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.optimize.low_memory_optimize import get_delta_params
from nuclear_qmc.diagnostic.plot_local_energy import plot_local_energy as plt_energy


def get_local_energy_for_block(
        psi_prefactor
        , psi_params
        , psi_vector
        , r_coords_for_block
        , hamiltonian
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
    local_energy_values = vmap(get_local_energy, in_axes=(None, None, None, 0, None))(psi_prefactor
                                                                                      , psi_params
                                                                                      , psi_vector
                                                                                      , r_coords_for_block
                                                                                      , hamiltonian)
    local_energy = local_energy_values.mean()
    return local_energy


def optimize_wave_function(
        n_proton
        , n_neutron
        , psi_prefactor
        , psi_params
        , psi_vector
        , psi_param_file
        , hamiltonian
        , seed=0
        , n_dimensions=3
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
        , plot_local_energy=False
        , local_energy_plot_limits=None

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
    hamiltonian: function
        Returns H|psi> given `psi_prefactor`, `psi_params`, `psi_vector`, and r_coords.
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
    logging.info("Search String | Step | Energy | Error")
    logging.info("------------- | ---- | ------ | -----")
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
                                          , in_axes=(None, None, None, 0, None))(psi_prefactor
                                                                                 , psi_params
                                                                                 , psi_vector
                                                                                 , r_coord_samples
                                                                                 , hamiltonian)
            local_energy = local_energy_per_block.mean()
            ddof = 1 if n_blocks > 1 else 0
            local_energy_error = jnp.std(local_energy_per_block, ddof=ddof)
            local_energy_error = local_energy_error / jnp.sqrt(n_blocks)
            logging.info(f'optimization step | {n_opt} | {local_energy} | {local_energy_error}')

        if plot_local_energy:
            plt_energy(psi_prefactor, psi_params, psi_vector, hamiltonian, r_coord_samples, local_energy_plot_limits,
                       f'local_energy_{n_opt:05}.png')

        # compute average wave function parameter update over each block
        def sum_delta_params(i, args):
            _delta_params_sum = args[0]
            _params = args[1]
            _delta_params_sum += get_delta_params(
                psi_prefactor
                , _params
                , psi_vector
                , r_coord_samples[i]
                , learning_rate
                , hamiltonian
                , return_loss=False
                , eps=epsilon_sr)
            return _delta_params_sum, _params

        args = (jnp.zeros_like(psi_params), psi_params)
        args = fori_loop(0, n_blocks, sum_delta_params, args)
        delta_params_avg = args[0] / n_blocks
        delta_params_avg = jnp.clip(delta_params_avg, -0.5, 0.5)
        psi_params += delta_params_avg
        jnp.save(psi_param_file, psi_params)

    return key, psi_params
