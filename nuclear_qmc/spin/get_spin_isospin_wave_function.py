import jax.numpy as jnp
from operator import add
from jax.ops import index_update, index
from jax import vmap, numpy as jnp
from copy import deepcopy

from scipy.special import comb
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import Permutation
from scipy.stats import rankdata
from jax.ops import index, index_update
from sympy import symbols, Matrix
import copy
import numpy as np
from nuclear_qmc.spin.get_tables import get_number_of_isospin_states, get_number_of_spin_states
from jax.ops import index, index_update
from itertools import permutations as get_permutations

from nuclear_qmc.spin.spherical_harmonics import get_spherical_harmonic_names, get_spherical_harmonic_functions, \
    get_spherical_harmonic_system, get_spherical_harmonic_systems
from nuclear_qmc.wave_function.combine_wave_functions import combine_wave_functions


def add_spin_str(state_str_list, spin='d'):
    for i in range(len(state_str_list)):
        state_str_list[i] = spin + state_str_list[i]
        spin = 'u' if spin == 'd' else 'd'
    return state_str_list


def add_particle_number(states):
    return [s + str(n) for s in states for n in range(len(states))]


def add_orbitals(states, orbitals, new_orbital_every_n_states=2):
    i_orbital = -1
    for i_state in range(len(states)):
        if i_state % new_orbital_every_n_states == 0:
            i_orbital += 1
        states[i_state] = orbitals[i_orbital] + states[i_state]
    return states


def get_permutation_signature(permutations):
    return jnp.array([Permutation(perm).signature() for perm in permutations])


def get_state_permutations(states, permutations, n_particles):
    representations = states[np.array(permutations)]
    # particle_numbers = np.arange(n_particles).astype(np.str)
    # representations = np.array([[elm+i for elm, i in zip(rep, particle_numbers)]for rep in representations])
    return np.array(representations)


def get_spin_and_isospin_indices(state_representations):
    iso_indices = []
    spin_indices = []
    for strs in state_representations:
        # # sort each product in term by particle number
        # strs = sorted(strs, key=lambda x: -int(x[-1]))
        # strs = [x[:-1] for x in strs]  # remove particle number from str

        # get iso spin indices
        iso_expr = ''
        for s in strs: iso_expr += s[-1]
        iso_index = iso_expr.replace('p', '1').replace('n', '0')
        iso_index = int(iso_index, 2)
        iso_indices.append(iso_index)

        # get spin indices
        spin_expr = ''
        for s in strs: spin_expr += s[-2]
        spin_index = spin_expr.replace('u', '1').replace('d', '0')
        spin_index = int(spin_index, 2)
        spin_indices.append(spin_index)

    # iso spin is condensed to charge conserved states
    iso_indices = rankdata(iso_indices, method='dense') - 1

    return jnp.array(spin_indices), jnp.array(iso_indices)


def get_orbitals(n_particle_of_type):
    n_pshell = get_number_of_p_shell_particles(n_particle_of_type, max_in_lower_shell=2)
    neutron_orbitals = ['S'] + n_pshell * ['P']
    return neutron_orbitals


def get_states_per_type(n_particle_of_type, iso_type_str):
    states = n_particle_of_type * [iso_type_str]
    states = add_spin_str(states)
    orbitals = get_orbitals(n_particle_of_type)
    states = add_orbitals(states, orbitals, new_orbital_every_n_states=2)
    return states


def get_states(n_protons, n_neutrons):
    p_states = get_states_per_type(n_protons, 'p')
    n_states = get_states_per_type(n_neutrons, 'n')
    states = n_states + p_states
    states = np.array(states)
    return states


def get_orbital_wave_function(function_permutations, functions, param_indices, spin_indices, iso_indices, n_particles,
                              n_spin, n_iso
                              , spin_isospin_signs):
    p_index = jnp.arange(n_particles)

    def func(params, r):
        params = params[param_indices]
        particle_function_matrix = jnp.array(
            [[f(p, ri) for f, p in zip(functions, params)] for ri in r])  # [n_particles, n_functions]
        evaluations = particle_function_matrix[p_index, function_permutations]  # [n_permutations, n_functions]
        psi = vmap(jnp.prod, in_axes=(0))(evaluations)
        out = jnp.ones((n_iso, n_spin), dtype=jnp.complex64)
        out = index_update(out, index[iso_indices, spin_indices], psi)
        out *= spin_isospin_signs
        return out

    return func


def get_wave_function_sign(n_spin_states, n_isospin_states, spin_indices, iso_indices, signatures, dtype):
    # build wfc assume all zeros and then fill in non zero elements using signatures from above
    wfc = np.zeros(shape=(n_isospin_states, n_spin_states))
    for iso, spin, sign in zip(iso_indices, spin_indices, signatures):
        wfc[iso, spin] = sign
    spin_isospin_signs = jnp.array(wfc, dtype=dtype)
    return spin_isospin_signs


def get_spin_isospin_indices(n_protons, n_neutrons):
    n_particles = n_protons + n_neutrons
    permutations = get_permutations(range(n_particles))
    permutations = jnp.array(list(permutations))
    states = get_states(n_protons, n_neutrons)
    state_permutations = get_state_permutations(states, permutations, n_particles)
    spin_indices, iso_indices = get_spin_and_isospin_indices(state_permutations)
    return spin_indices, iso_indices, permutations


def get_number_of_p_shell_particles(n_particles, max_in_lower_shell=4):
    n_p = n_particles - max_in_lower_shell
    if n_p <= 0:
        return 0
    elif n_p > max_in_lower_shell:
        return max_in_lower_shell
    else:
        return n_p


def get_number_of_orbital_configurations(n_particles):
    if n_particles > 8:
        raise RuntimeError('number of orbital configs only works for n particles <= 8')
    n_p_shell_particles = get_number_of_p_shell_particles(n_particles)
    return comb(n_particles, n_p_shell_particles)


def get_indices(state_permutations, state_position_index, str_to_int_rules, use_rank=True):
    # extract state str at given index
    original_shape = state_permutations.shape
    states = np.array([str_to_int_rules[e[state_position_index]] for l in state_permutations for e in l]).reshape(
        *original_shape)
    states = [''.join(lst) for lst in states]
    if use_rank:
        states = [int(str) for str in states]
        states = rankdata(states, method='dense') - 1
    else:
        states = [int(str, 2) for str in states]
    return jnp.array(states)


def build_spin_isospin_system(n_neutrons, n_protons):
    # build initial wave function
    n_particles = n_neutrons + n_protons
    n_iso_configs = get_number_of_isospin_states(n_particles, n_protons)
    n_spin_configs = get_number_of_spin_states(n_particles)
    n_orbital_configs = int(get_number_of_orbital_configurations(n_particles))
    wave_function = jnp.zeros(shape=(n_orbital_configs, n_iso_configs, n_spin_configs))

    # get indices for permutation signatures
    coordinate_permutations = jnp.array(list(get_permutations(range(n_particles))))
    states = get_states(n_protons, n_neutrons)
    state_permutations = get_state_permutations(states, coordinate_permutations, n_particles)
    orbital_indices = get_indices(state_permutations, 0, {'S': '0', 'P': '1'}, use_rank=True)
    spin_indices = get_indices(state_permutations, 1, {'d': '0', 'u': '1'}, use_rank=False)
    iso_indices = get_indices(state_permutations, 2, {'n': '0', 'p': '1'}, use_rank=True)
    indices = jnp.stack((orbital_indices, iso_indices, spin_indices), axis=-1)

    # fill wfc with values that enforce antisymmetry (permutation signatures)
    permutation_signatures = jnp.array([Permutation(perm).signature() for perm in coordinate_permutations])
    wave_function = index_update(wave_function, index[indices[:, 0], indices[:, 1], indices[:, 2]],
                                 permutation_signatures)
    return wave_function, indices, coordinate_permutations
