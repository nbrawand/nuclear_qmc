from jax.config import config
import jax.numpy as jnp
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian.get_local_energy import get_local_energy
from nuclear_qmc.operators.hamiltonian.build_hamiltonian import build_hamiltonian
from nuclear_qmc.wave_function.legacy_wave_function_for_testing.test_neural_network import build_test_nn_wfc
from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.wave_function.utility import get_wave_function_system

config.update("jax_enable_x64", True)


def test_full_energy_run():
    N_PROTON = 1
    N_NEUTRON = 1
    INITIAL_WALKER_STANDARD_DEVIATION = 1.0
    WALKER_STEP_SIZE = 1.0
    N_WALKERS = 2000
    N_DIMENSIONS = 3
    N_EQUILIBRIUM_STEPS = 100
    N_STEPS = 20
    N_VOID_STEPS = 100
    _, psi_prefactor, psi_params = build_test_nn_wfc()
    particle_pairs, particle_triplets, psi_vector, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
        N_PROTON, N_NEUTRON,
        dtype=jnp.float64,
        as_jax_array=True)
    hamiltonian = build_hamiltonian(psi_vector, 'arxiv_2007_14282v2', particle_pairs, particle_triplets,
                                    spin_exchange_indices,
                                    isospin_exchange_indices)
    key = random.PRNGKey(0)
    key, r_coord_samples = sample(psi_prefactor
                                  , psi_params
                                  , psi_vector
                                  , N_STEPS
                                  , WALKER_STEP_SIZE
                                  , N_WALKERS
                                  , N_NEUTRON + N_PROTON
                                  , N_DIMENSIONS
                                  , N_EQUILIBRIUM_STEPS
                                  , N_VOID_STEPS
                                  , key
                                  , INITIAL_WALKER_STANDARD_DEVIATION
                                  )

    r_coords = r_coord_samples.reshape(-1, N_PROTON + N_NEUTRON, N_DIMENSIONS)
    local_energy = vmap(get_local_energy, in_axes=(None, None, None, 0, None))(psi_prefactor
                                                                               , psi_params
                                                                               , psi_vector
                                                                               , r_coords
                                                                               , hamiltonian)
    computed = local_energy.mean().round(8)
    expected = jnp.array(-2.20467771, dtype=jnp.float64).round(8)  # changed after updating mass and hbar
    assert jnp.array_equal(expected, computed)
