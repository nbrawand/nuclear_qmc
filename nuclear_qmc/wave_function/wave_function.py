import jax.numpy as jnp
from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
from nuclear_qmc.spin.get_tables import get_spin_particle_pairs, get_spin_exchange_indices, get_isospin_exchange_index, \
    get_spin_state_indices
from nuclear_qmc.utils.get_triplets import get_triplets


def get_wave_function_system(n_protons, n_neutrons, dtype=jnp.float64, as_jax_array=True):
    mass_number = n_protons + n_neutrons
    particle_pairs = get_spin_particle_pairs(mass_number, as_jax_array)
    particle_triplets = get_triplets(jnp.arange(mass_number))
    spin_exchange_indices = get_spin_exchange_indices(particle_pairs
                                                      , get_spin_state_indices(mass_number, as_jax_array)
                                                      , as_jax_array)
    spin = get_spin_isospin_wave_function(n_protons, n_neutrons, dtype=dtype)
    isospin_exchange_indices = get_isospin_exchange_index(particle_pairs
                                                          , mass_number
                                                          , n_protons
                                                          , as_jax_array
                                                          , also_return_binary_representation=False)
    return particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices


def get_psi_r(psi_prefactor, psi_parameters, r_coords, psi_vector):
    psi_r = psi_prefactor(psi_parameters, r_coords)
    psi_r *= psi_vector
    return psi_r
