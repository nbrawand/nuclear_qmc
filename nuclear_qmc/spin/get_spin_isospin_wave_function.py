import jax.numpy as jnp
from scipy.stats import rankdata
from sympy import symbols, Matrix
import copy
import numpy as np
from nuclear_qmc.spin.get_tables import get_number_of_isospin_states, get_number_of_spin_states
from jax.ops import index, index_update


def add_spin_str(state_str_list, spin='d'):
    for i in range(len(state_str_list)):
        state_str_list[i] = spin + state_str_list[i]
        spin = 'u' if spin == 'd' else 'd'
    return state_str_list


def add_particle_number(states):
    return [s + str(n) for s in states for n in range(len(states))]



def get_spin_isospin_wave_function(n_protons, n_neutrons, dtype=jnp.float64):
    # build symbolic states to take det and manipulate str rep
    p_states = n_protons * ['p']
    p_states = add_spin_str(p_states)
    n_states = n_neutrons * ['n']
    n_states = add_spin_str(n_states)
    states = p_states + n_states
    states = add_particle_number(states)
    states = symbols(states)

    # build matrix
    n_particles = n_protons + n_neutrons
    matrix = np.array(states).reshape(-1, n_particles)
    mat_det_terms = Matrix(matrix).det().args

    # build list of signs and spin & isospin indices for each element from det
    iso_indices = []
    signs = []
    spin_indices = []
    for term in mat_det_terms:
        # grab sign from each term in det
        sign = -1 if term.args[0] == -1 else +1
        signs.append(sign)

        # sort each product in term by particle number
        strs = [str(x) for x in term.args]
        strs = strs[1:] if sign < 0 else strs
        strs = sorted(strs, key=lambda x: -int(x[-1]))
        strs = [x[:-1] for x in strs]  # remove particle number from str

        # get iso spin indices
        iso_expr = ''
        for s in strs: iso_expr += s[1]
        iso_index = iso_expr.replace('p', '1').replace('n', '0')
        iso_index = int(iso_index, 2)
        iso_indices.append(iso_index)
        spin_expr = ''

        # get spin indices
        for s in strs: spin_expr += s[0]
        spin_index = spin_expr.replace('u', '1').replace('d', '0')
        spin_index = int(spin_index, 2)
        spin_indices.append(spin_index)

    # iso spin is condensed to charge conserved states
    iso_indices = rankdata(iso_indices, method='dense') - 1

    # build wfc assume all zeros and then fill in non zero elements using signs from above
    n_spin_states = get_number_of_spin_states(n_particles)
    n_isospin_states = get_number_of_isospin_states(n_particles, n_protons)
    wfc = np.zeros(shape=(n_isospin_states, n_spin_states))
    for iso, spin, sign in zip(iso_indices, spin_indices, signs):
        wfc[iso, spin] = sign
    return jnp.array(wfc, dtype=dtype)
