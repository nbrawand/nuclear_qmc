import jax.numpy as jnp
from sympy.combinatorics.permutations import Permutation
import numpy as np
from itertools import permutations as get_permutations

from nuclear_qmc.spin.get_tables import get_number_of_spin_states, get_number_of_isospin_states


def build_spin_isospin(n_particles, n_protons, iso_indices, spin_indices, permutations):
    # compute signatures of determinant using permutations (see Leibniz formula)
    signatures = jnp.array([Permutation(perm).signature() for perm in permutations])

    # build spin-isospin vector assume all zeros and then fill in non zero elements using signatures from above
    n_spin_states = get_number_of_spin_states(n_particles)
    n_isospin_states = get_number_of_isospin_states(n_particles, n_protons)
    wfc = np.zeros(shape=(n_isospin_states, n_spin_states), dtype=jnp.float64)
    wfc[iso_indices, spin_indices] = signatures
    signatures = jnp.array(wfc, dtype=jnp.float64)

    def wave_function(parameters, r_coords, current_wave_function=1.0, parameter_index=0):
        current_wave_function = signatures
        return parameters, r_coords, current_wave_function, parameter_index

    return wave_function
