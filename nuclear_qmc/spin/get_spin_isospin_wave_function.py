import jax.numpy as jnp
from operator import add
from jax import vmap
from copy import deepcopy
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
    return representations


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


def get_states(n_protons, n_neutrons):
    p_states = n_protons * ['p']
    p_states = add_spin_str(p_states)
    # if proton_orbitals is not None:
    #    p_states = add_orbitals(p_states, proton_orbitals, new_orbital_every_n_states=2)
    n_states = n_neutrons * ['n']
    n_states = add_spin_str(n_states)
    # if neutron_orbitals is not None:
    #    n_states = add_orbitals(n_states, neutron_orbitals, new_orbital_every_n_states=2)
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
