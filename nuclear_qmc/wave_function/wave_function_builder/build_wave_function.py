import re
import copy
from jax import jit

from nuclear_qmc.wave_function.partition_jastro.partition_jastro import get_partition_jastro
from nuclear_qmc.wave_function.wave_function_builder.spherical_harmonics import Y1m1, Y10, Y11, build_radial_function, \
    SPHERICAL_HARMONICS
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
    orbitals = get_orbitals(n_particle_of_type)
    states = add_orbitals(states, orbitals, new_orbital_every_n_states=2)
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
    replace_state = lambda x: str_to_int_rules[x.split('_')[state_position_index]]
    states = apply_func(state_permutations, replace_state)
    states = [''.join(lst) for lst in states]
    if use_rank:
        states = [int(str) for str in states]
        states = rankdata(states, method='dense') - 1
    else:
        states = [int(str, 2) for str in states]
    return jnp.array(states)


def build_slice_func(func, start, stop):
    def f(x, y):
        return func(x[start:stop], y)

    return f


def create_wave_function(key
                         , n_iso_configs
                         , n_spin_configs
                         , state_permutations
                         , signature_indices
                         , permutation_signatures
                         , n_dense
                         , n_hidden_layers
                         , coefficients
                         , add_partition_jastro=False
                         , confining_factor=0.1
                         ):
    n_particles = state_permutations.shape[-1]
    params = jnp.array([])
    signature_indices = signature_indices.reshape(-1, 2)
    coefficients = jnp.array(coefficients).reshape(-1, 1)

    @jit
    def accumulate_wave_function(terms):
        wave_function = jnp.zeros(shape=(n_iso_configs, n_spin_configs))
        terms = terms.reshape(-1)
        i = signature_indices[:, 0]
        j = signature_indices[:, 1]
        wave_function = index_add(wave_function, index[i, j], terms)
        return wave_function

    def decay_func(_r, decay_strength):
        mag_r = jnp.linalg.norm(_r) ** 2
        return jnp.exp(-decay_strength * mag_r)

    # Create orbitals, only works for Li
    # replace the Pneutron->A and Pproton->B
    orbitals = apply_func(state_permutations, lambda x: '_'.join(x.split('_')[:2]))

    unique_orbitals = np.unique(orbitals)

    functions = []
    radial_functions = []
    function_param_start_stop = []
    seen_radial_orbitals = []
    function_dict = {}
    i = 0

    def make_function(indx, start, stop, y):
        return lambda _p, _r: radial_functions[indx](_p[start:stop], _r) * SPHERICAL_HARMONICS[y](_r) * decay_func(_r,
                                                                                                                   confining_factor)

    for orbital in unique_orbitals:
        r = orbital.split('_')[0]
        if r not in seen_radial_orbitals:
            if r == '1':
                radial_func, radial_params = lambda _p, _r: 1.0, jnp.array([])
            else:
                key, radial_func, radial_params = build_radial_function(key
                                                                        , n_dense
                                                                        , n_hidden_layers
                                                                        , nn_wrapper_function=jnp.exp)
            radial_functions.append(radial_func)
            start = len(params)
            stop = start + len(radial_params)
            function_param_start_stop.append([start, stop])
            params = jnp.concatenate((params, radial_params))
            seen_radial_orbitals.append(r)

        indx = seen_radial_orbitals.index(r)
        start, stop = function_param_start_stop[indx]

        y = orbital.split('_')[-1]
        functions.append(make_function(indx, start, stop, y))
        function_dict[orbital] = i
        i += 1

    function_indices = jnp.array([apply_func(orbs, lambda x: function_dict[x]) for orbs in orbitals])

    particles = jnp.arange(n_particles)

    n_orbital_params = len(params)

    if add_partition_jastro:
        get_r_orbs = lambda x: x.split('_')[0]
        partition_psis = []
        for state_perms_per_det in state_permutations:
            key, partition_psi, partition_params = get_partition_jastro(key
                                                                        , apply_func(state_perms_per_det, get_r_orbs)
                                                                        , n_dense
                                                                        , n_hidden_layers, latent_shape=4, debug=False)
            start = len(params)
            params = jnp.concatenate((params, partition_params))
            stop = len(params)
            func = build_slice_func(partition_psi, start, stop)
            partition_psis.append(func)

    @jit
    def psi(in_params, r_coords):
        # apply every function to each particle coordinate
        orbital_i_r_j = jnp.array(
            [vmap(func, in_axes=(None, 0))(in_params[:n_orbital_params], r_coords) for func in
             functions])  # n_functions, n_particles
        # expand orbital values for each permutation
        orbitals = orbital_i_r_j[
            function_indices[:, :, particles], particles]  # n_determinant, n_permutation, n_particles
        # multiply all orbitals together
        orbitals = jnp.prod(orbitals, axis=-1)  # n_determinant, n_permutation
        if add_partition_jastro:
            partition_jastro = jnp.array([p_psi(in_params, r_coords).reshape(-1) for p_psi in partition_psis])
            orbitals = orbitals * partition_jastro
        orbitals = coefficients * orbitals * permutation_signatures
        wave_function = accumulate_wave_function(orbitals)
        return wave_function

    return key, psi, params


def build_wave_function(key, n_neutron, n_proton, n_dense, n_hidden_layers, states, coefficients
                        , add_partition_jastro=False, confining_factor=0.1):
    # build initial wave function
    n_particles = n_neutron + n_proton
    n_iso_configs = get_number_of_isospin_states(n_particles, n_proton)
    n_spin_configs = get_number_of_spin_states(n_particles)

    # get indices for permutation signatures
    coordinate_permutations = jnp.array(list(get_permutations(range(n_particles))))
    # states = get_states(n_proton, n_neutron)
    states = np.array(states)
    state_permutations = np.array([get_state_permutations(s, coordinate_permutations, n_particles) for s in states])

    # orbital_indices = get_indices(state_permutations, 0, {'S': '0', 'P': '1'}, use_rank=True)
    spin_indices = np.array([get_indices(s, -2, {'d': '0', 'u': '1'}, use_rank=False) for s in state_permutations])
    iso_indices = np.array([get_indices(s, -1, {'n': '0', 'p': '1'}, use_rank=True) for s in state_permutations])
    indices = jnp.stack((iso_indices, spin_indices), axis=-1)

    # fill wfc with values that enforce antisymmetry (permutation signatures)
    permutation_signatures = jnp.array([Permutation(perm).signature() for perm in coordinate_permutations])

    # orbital function
    key, psi, psi_params = create_wave_function(key
                                                , n_iso_configs
                                                , n_spin_configs
                                                , state_permutations
                                                , indices
                                                , permutation_signatures
                                                , n_dense
                                                , n_hidden_layers
                                                , coefficients
                                                , add_partition_jastro
                                                , confining_factor
                                                )

    return key, psi, psi_params
