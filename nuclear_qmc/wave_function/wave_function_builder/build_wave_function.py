import re
from jax import jit
from nuclear_qmc.wave_function.wave_function_builder.spherical_harmonics import Y1m1, Y10, Y11, SPHERICAL_HARMONICS
from nuclear_qmc.wave_function.wave_function_builder.build_radial_functions import build_radial_function
from jax import vmap, numpy as jnp
from sympy.combinatorics.permutations import Permutation
from scipy.stats import rankdata
import numpy as np
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_spin_isospin_indices import get_number_of_isospin_states, \
    get_number_of_spin_states
from jax.ops import index
from itertools import permutations as get_permutations
from jax.ops import index_add
from nuclear_qmc.wave_function.utility import apply_confining_potential


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


def get_state_permutations(states, permutations, n_particles):
    representations = states[np.array(permutations)]
    # particle_numbers = np.arange(n_particles).astype(np.str)
    # representations = np.array([[elm+i for elm, i in zip(rep, particle_numbers)]for rep in representations])
    return np.array(representations)


def get_orbitals(n_particle_of_type):
    n_pshell = get_number_of_p_shell_particles(n_particle_of_type, max_in_lower_shell=2)
    neutron_orbitals = ['S'] + n_pshell * ['P']
    return neutron_orbitals


def get_states_per_type(n_particle_of_type, iso_type_str):
    states = n_particle_of_type * [iso_type_str]
    states = add_spin_str(states)
    # orbitals = get_orbitals(n_particle_of_type)
    # states = add_orbitals(states, orbitals, new_orbital_every_n_states=2)
    return states


def get_states(n_protons, n_neutrons):
    p_states = get_states_per_type(n_protons, 'p')
    n_states = get_states_per_type(n_neutrons, 'n')
    states = n_states + p_states
    states = np.array(states)
    return states


def get_number_of_p_shell_particles(n_particles, max_in_lower_shell=4):
    n_p = n_particles - max_in_lower_shell
    if n_p <= 0:
        return 0
    elif n_p > max_in_lower_shell:
        return max_in_lower_shell
    else:
        return n_p


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


def create_wave_function(key
                         , n_iso_configs
                         , n_spin_configs
                         , coordinate_permutations
                         , signature_indices
                         , permutation_signatures
                         , n_dense
                         , n_hidden_layers
                         , angular_orbitals
                         , radial_orbitals
                         , coefficients
                         ):
    n_particles = coordinate_permutations.shape[-1]

    @jit
    def accumulate_wave_function(terms):
        wave_function = jnp.zeros(shape=(n_iso_configs, n_spin_configs))
        i = signature_indices[:, 0]
        j = signature_indices[:, 1]
        wave_function = index_add(wave_function, index[i, j], terms)
        return wave_function

    # create radial functions
    radial_functions = {}
    radial_functions_param_indices = {}
    params = jnp.array([])
    for radial_func_name in set([elm for lst in radial_orbitals for elm in lst]):
        if radial_func_name == 0:
            radial_func = lambda _p, _r: 1.0
            radial_params = jnp.array([])
        else:
            key, radial_func, radial_params = build_radial_function(key
                                                                    , n_dense
                                                                    , n_hidden_layers
                                                                    , nn_wrapper_function=jnp.exp)
        radial_functions[radial_func_name] = radial_func
        radial_functions_param_indices[radial_func_name] = [len(params), len(params) + len(radial_params)]
        params = jnp.concatenate((params, radial_params))

    def decay_func(_r, decay_strength):
        mag_r = jnp.linalg.norm(_r) ** 2
        return jnp.exp(-decay_strength * mag_r)

    def get_f_key(r_orb, y_orb):
        return str(r_orb) + '_' + str(y_orb)

    # build orbital functions R*Y
    functions = []
    function_index = {}
    i = 0
    for r_orbs, y_orbs in zip(radial_orbitals, angular_orbitals):
        for r_orb, y_orb in zip(r_orbs, y_orbs):
            f_key = get_f_key(r_orb, y_orb)
            if f_key not in function_index.keys():
                angular_func = SPHERICAL_HARMONICS[y_orb]
                p_start, p_end = radial_functions_param_indices[r_orb]
                functions.append(
                    lambda _p, _r: radial_func(_p[p_start:p_end], _r) * angular_func(_r) * decay_func(_r, 0.1)
                )
                function_index[f_key] = i
                i += 1

    # create list of lists of function indices that match those in the radial and angular orbitals lists
    function_indices = []
    for r_orbs, y_orbs in zip(radial_orbitals, angular_orbitals):
        tmp = []
        for r_orb, y_orb in zip(r_orbs, y_orbs):
            f_key = get_f_key(r_orb, y_orb)
            tmp.append(function_index[f_key])
        function_indices.append(tmp)
    function_indices = jnp.array(function_indices, dtype=jnp.int32)

    # expand function indices using the coordinate permutations
    # n_det, n_particles
    function_indices = function_indices[:, coordinate_permutations]

    coefficients = jnp.array(coefficients, dtype=jnp.float64)
    particles = jnp.arange(n_particles)

    def psi(in_params, r_coords):
        # apply every function to each particle coordinate
        orbital_i_r_j = jnp.array(
            [vmap(func, in_axes=(None, 0))(in_params, r_coords) for func in functions])  # n_functions, n_particles
        # expand orbital values for each permutation
        orbitals = orbital_i_r_j[
            function_indices[:, :, particles], particles]  # n_determinant, n_permutation, n_particles
        # multiply all orbitals together
        orbitals = jnp.prod(orbitals, axis=-1)  # n_determinant, n_permutation
        # sum over determinants
        orbitals = jnp.einsum('i,ij', coefficients, orbitals)  # n_permutation
        orbitals *= permutation_signatures
        wave_function = accumulate_wave_function(orbitals)
        return wave_function

    return key, psi, params


def build_wave_function(key, n_neutron, n_proton, n_dense, n_hidden_layers
                        , radial_orbitals
                        , angular_orbitals
                        , coefficients
                        ):
    # build initial wave function
    n_particles = n_neutron + n_proton
    n_iso_configs = get_number_of_isospin_states(n_particles, n_proton)
    n_spin_configs = get_number_of_spin_states(n_particles)

    # get indices for permutation signatures
    coordinate_permutations = jnp.array(list(get_permutations(range(n_particles))))
    states = get_states(n_proton, n_neutron)
    state_permutations = get_state_permutations(states, coordinate_permutations, n_particles)

    # orbital_indices = get_indices(state_permutations, 0, {'S': '0', 'P': '1'}, use_rank=True)
    spin_indices = get_indices(state_permutations, 0, {'d': '0', 'u': '1'}, use_rank=False)
    iso_indices = get_indices(state_permutations, 1, {'n': '0', 'p': '1'}, use_rank=True)
    indices = jnp.stack((iso_indices, spin_indices), axis=-1)

    # fill wfc with values that enforce antisymmetry (permutation signatures)
    permutation_signatures = jnp.array([Permutation(perm).signature() for perm in coordinate_permutations])

    # orbital function
    key, psi, psi_params = create_wave_function(key
                                                , n_iso_configs
                                                , n_spin_configs
                                                , coordinate_permutations
                                                , indices
                                                , permutation_signatures
                                                , n_dense
                                                , n_hidden_layers
                                                , angular_orbitals
                                                , radial_orbitals
                                                , coefficients
                                                )

    return key, psi, psi_params
