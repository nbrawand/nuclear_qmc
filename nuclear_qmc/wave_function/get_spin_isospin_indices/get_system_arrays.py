from jax import numpy as jnp
from scipy.stats import rankdata
from math import factorial
import numpy as np

from nuclear_qmc.utils.get_triplets import get_triplets


def get_number_of_spin_states(n_particles):
    return 2 ** n_particles


def get_spin_state_indices(mass_number, as_jax_array=True):
    package = jnp if as_jax_array else np
    return package.array(package.arange(get_number_of_spin_states(mass_number)), dtype=package.int32)


def get_spin_particle_pairs(mass_number, as_jax_array=True):
    """Returns the indices for each particle pair.

    Examples
    --------
    >>>get_spin_particle_pairs(3)
   [[0, 1]
    [0, 2]
    [1, 2]]
    """
    package = jnp if as_jax_array else np
    return package.array([[p1, p2] for p2 in range(1, mass_number) for p1 in range(p2)], dtype=package.int32)


def get_spin_exchange_index(state_index, particle_pair, package):
    particle_index_value = 2 ** particle_pair
    binary_digit = state_index >> particle_pair.reshape(-1, 1) & 1
    exchanged_index = state_index - package.dot(particle_index_value, binary_digit)
    exchanged_index += package.dot(particle_index_value, binary_digit[::-1])
    return exchanged_index


def get_spin_exchange_indices(particle_pairs, state_indices, as_jax_array=True):
    """These are the wave function indices after exchanging the spin for each particle pair.

    Returns
    -------
        Returned array is of dimensions [n_pair, n_isospin, n_spin] and is used by the sigma operator.
    """
    package = jnp if as_jax_array else np
    pairs_indices_in_binary_rep = convert_particle_index_to_binary_index(particle_pairs, package)
    return package.array([get_spin_exchange_index(state_indices, particle_pair, package)
                          for particle_pair in pairs_indices_in_binary_rep], dtype=package.int32).T


def get_number_of_isospin_states(mass_number, proton_number):
    mass_factorial = factorial(mass_number)
    proton_factorial = factorial(proton_number)
    neutron_factorial = factorial(mass_number - proton_number)
    return int(mass_factorial / proton_factorial / neutron_factorial)


def get_isospin_state_indices(mass_number, proton_number, as_jax_array=True):
    package = jnp if as_jax_array else np
    number_of_isospin_states = get_number_of_isospin_states(mass_number, proton_number)
    return package.array(package.arange(number_of_isospin_states), dtype=package.int32)


def convert_particle_index_to_binary_index(arr, package):
    max_particle_index = arr.max()
    return package.abs(arr - max_particle_index)


def get_raw_spin_indices(mass_number, as_jax_array):
    if mass_number > 64:
        raise RuntimeError(
            'get_isospin_exchange_index: binary representation limited to 64 digits mass number must be <= 64.')
    package = jnp if as_jax_array else np
    if mass_number < 9:
        int_size = package.uint8
    elif mass_number < 17:
        int_size = package.uint16
    elif mass_number < 33:
        int_size = package.uint32
    else:
        int_size = package.uint64
    indices = package.arange(2 ** mass_number, dtype=int_size)
    return indices


def get_raw_isospin_indices(mass_number, proton_number, as_jax_array=True):
    package = jnp if as_jax_array else np
    indices = get_raw_spin_indices(mass_number, as_jax_array)
    valid_binary_representation = package.unpackbits(indices.reshape(-1, 1).view(package.uint8)
                                                     , axis=1)  # bits out of order from view(8) but the sum is correct
    number_of_protons = valid_binary_representation.sum(axis=1)
    number_of_isospin_states = get_number_of_isospin_states(mass_number, proton_number)
    valid_isospin_indices = indices[number_of_protons == proton_number][:number_of_isospin_states]
    valid_binary_representation = valid_binary_representation[valid_isospin_indices]
    return valid_binary_representation, valid_isospin_indices


def get_isospin_exchange_index(particle_pairs, mass_number, proton_number, as_jax_array=True,
                               also_return_binary_representation=False):
    """Returns the wave function indices after exchanging the isospin for each particle pair.

    Returns
    -------
        The returned has dimensions [n_pair, n_spin, n_isospin] and is used by the tau operator.
    """
    package = jnp if as_jax_array else np
    valid_binary_representation, valid_isospin_indices = get_raw_isospin_indices(mass_number,
                                                                                 proton_number, as_jax_array=True)
    #  calculate indices for valid isospin states
    exchange_indices = get_spin_exchange_indices(particle_pairs, valid_isospin_indices, as_jax_array=as_jax_array)
    exchange_indices = rankdata(exchange_indices, method='dense').reshape(exchange_indices.shape) - 1

    if also_return_binary_representation:
        #  reorganize the 8 bit chunks from view(8) call if necessary
        tot_len = valid_binary_representation.shape[-1]
        number_of_8_bit_chunks = (valid_binary_representation.shape[-1] + 1) // 8
        work_array = valid_binary_representation[:, tot_len - 8:tot_len].copy()
        for i_chunk in range(1, number_of_8_bit_chunks):
            stop = tot_len - i_chunk * 8
            start = stop - 8
            work_array = package.concatenate((work_array, valid_binary_representation[:, start:stop]), axis=1)
        valid_binary_representation = work_array[:, -mass_number:]  # cut leading zeros
        return package.array(exchange_indices), valid_binary_representation
    else:
        return package.array(exchange_indices)


def get_system_arrays(n_protons, n_neutrons, as_jax_array=True,
                      also_return_binary_representation=False):
    """Helper function returns necessary arrays associated with the nuclear system. See called functions for details."""
    mass_number = n_protons + n_neutrons
    particle_pairs = get_spin_particle_pairs(mass_number, as_jax_array)
    particle_triplets = get_triplets(jnp.arange(mass_number))
    spin_exchange_indices = get_spin_exchange_indices(particle_pairs
                                                      , get_spin_state_indices(mass_number, as_jax_array)
                                                      , as_jax_array)
    # spin = get_spin_isospin_wave_function(n_protons, n_neutrons, dtype=dtype)
    isospin_exchange_indices = get_isospin_exchange_index(particle_pairs
                                                          , mass_number
                                                          , n_protons
                                                          , as_jax_array
                                                          ,
                                                          also_return_binary_representation=also_return_binary_representation)
    if also_return_binary_representation:
        isospin_exchange_indices, isospin_binary_representation = isospin_exchange_indices
        return particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices, isospin_binary_representation
    else:
        return particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices
