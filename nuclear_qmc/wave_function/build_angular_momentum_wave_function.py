from jax import vmap
from jax.ops import index, index_update
import jax.numpy as jnp
from nuclear_qmc.spin.spherical_harmonics import get_spherical_harmonic_functions


def build_angular_momentum_wave_function(n_particles
                                         , spherical_harmonic_function_names
                                         , function_permutations
                                         , iso_indices
                                         , spin_indices):
    spherical_harmonics = get_spherical_harmonic_functions(spherical_harmonic_function_names)
    particle_indices = jnp.arange(n_particles)

    def wave_function(parameters, r_coords, current_wave_function, parameter_index=0):
        y_r_matrix = jnp.array([[f(r) for f in spherical_harmonics] for r in r_coords],
                               dtype=jnp.complex64)  # [n_particles, n_func]
        y_r_vector = y_r_matrix[particle_indices, function_permutations]  # [n_permutations, n_functions]
        y_r_vector = vmap(jnp.prod)(y_r_vector)
        new_values = y_r_vector * current_wave_function[iso_indices, spin_indices]
        current_wave_function = index_update(current_wave_function, index[iso_indices, spin_indices], new_values)
        return parameters, r_coords, current_wave_function, parameter_index

    return wave_function
