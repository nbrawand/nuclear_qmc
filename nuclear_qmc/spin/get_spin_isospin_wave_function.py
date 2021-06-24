import jax.numpy as jnp
import re
from nuclear_qmc.spin.spherical_harmonics import Y1m1, Y10, Y11, build_radial_function
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


def apply_func(matrix, func):
    original_shape = matrix.shape
    matrix = matrix.reshape(-1)
    matrix = np.array(list(map(func, matrix)))
    matrix = matrix.reshape(*original_shape)
    return matrix


def get_indices(state_permutations, state_position_index, str_to_int_rules, use_rank=True):
    # extract state str at given index
    replace_state = lambda x: str_to_int_rules[x[state_position_index]]
    states = apply_func(state_permutations, replace_state)
    states = [''.join(lst) for lst in states]
    if use_rank:
        states = [int(str) for str in states]
        states = rankdata(states, method='dense') - 1
    else:
        states = [int(str, 2) for str in states]
    return jnp.array(states)


def build_orbital_wave_function(key, state_permutations, n_dense, n_hidden_layers, orbital_index=0):
    n_particles = state_permutations.shape[-1]

    if n_particles <= 4:
        # just S wave return 1.0
        return key, lambda p, r: 1.0, jnp.array([])
    elif n_particles == 6:
        # Create orbitals, only works for Li
        # replace the Pneutron->A and Pproton->B
        state_permutations = apply_func(state_permutations, lambda x: re.sub('P.n', 'A', x))
        state_permutations = apply_func(state_permutations, lambda x: re.sub('P.p', 'B', x))
        # strip just the orbital information
        state_permutations = apply_func(state_permutations, lambda x: x[orbital_index])

        # create radial functions
        key, radial_func, params = build_radial_function(key, n_dense, n_hidden_layers, nn_wrapper_function=jnp.exp)

        # setup orbital functions and indices to replace characters
        functions = [lambda p, r: 1.0
            , lambda p, r: radial_func(p, r) * Y11(r)
            , lambda p, r: radial_func(p, r) * Y10(r)
            , lambda p, r: radial_func(p, r) * Y1m1(r)]
        p_orbital_indices = [['1', '3'], ['2', '2'], ['3', '2']]  # 1,-1  0,0  -1,1
        sqrt3 = 1. / jnp.sqrt(3.)
        coef = jnp.array([sqrt3, -sqrt3, sqrt3], dtype=jnp.float64)  # 3 coefficients for each determinant

        # Replace orbital characters with indices A->p1, B->p2, S->0
        funcs = [lambda x: x.replace('A', p1).replace('B', p2).replace('S', '0') for p1, p2 in p_orbital_indices]
        # n_determinant, n_permutation, n_particles
        function_indices = np.array([apply_func(state_permutations.copy()
                                                , lambda x: x.replace('A', p1).replace('B', p2).replace('S', '0')) for
                                     p1, p2 in p_orbital_indices])
        function_indices = apply_func(function_indices, lambda x: int(x))
        function_indices = jnp.array(function_indices)

        particles = jnp.arange(n_particles)

        def psi(in_params, r_coords):
            # apply every function to each particle coordinate
            orbital_i_r_j = jnp.array(
                [vmap(func, in_axes=(None, 0))(in_params, r_coords) for func in functions])  # n_functions, n_particles
            # expand orbital values for each permutation
            out = orbital_i_r_j[
                function_indices[:, :, particles], particles]  # n_determinant, n_permutation, n_particles
            # multiply all orbitals together
            out = jnp.prod(out, axis=-1)  # n_determinant, n_permutation
            # sum over determinants
            out = jnp.einsum('i,ij', coef, out)  # n_permutation
            return out

        return key, psi, params
    else:
        raise RuntimeError('build_orbital_wave_function requires n_particles <= 4 or == 6')


def build_spin_isospin_system(key, n_neutrons, n_protons, n_dense, n_hidden_layers):
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

    # orbital function
    psi, psi_params = build_orbital_wave_function(key, state_permutations, n_dense, n_hidden_layers, orbital_index=0)

    orbital_indices = get_indices(state_permutations, 0, {'S': '0', 'P': '1'}, use_rank=True)
    spin_indices = get_indices(state_permutations, 1, {'d': '0', 'u': '1'}, use_rank=False)
    iso_indices = get_indices(state_permutations, 2, {'n': '0', 'p': '1'}, use_rank=True)
    indices = jnp.stack((orbital_indices, iso_indices, spin_indices), axis=-1)

    # fill wfc with values that enforce antisymmetry (permutation signatures)
    permutation_signatures = jnp.array([Permutation(perm).signature() for perm in coordinate_permutations])
    wave_function = index_update(wave_function, index[indices[:, 0], indices[:, 1], indices[:, 2]],
                                 permutation_signatures)

    return key, wave_function, indices, psi, psi_params
