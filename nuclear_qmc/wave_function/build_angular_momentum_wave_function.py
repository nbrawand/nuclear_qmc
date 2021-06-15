from jax import vmap, jit
from jax.ops import index, index_update
import jax.numpy as jnp
from nuclear_qmc.spin.spherical_harmonics import get_spherical_harmonic_functions, get_spherical_harmonic_systems


def build_angular_momentum_wave_function(key, n_particles
                                         , function_permutations
                                         , iso_indices
                                         , spin_indices
                                         , L_total
                                         , L_z_total
                                         , L_1
                                         , L_2
                                         , spin_isospin_wave_function
                                         , n_dense
                                         , n_hidden_layers
                                         ):
    key, coefficients, spherical_harmonics_list, params = get_spherical_harmonic_systems(key, n_particles
                                                                                         , L_total
                                                                                         , L_z_total
                                                                                         , L_1
                                                                                         , L_2, n_dense,
                                                                                         n_hidden_layers)
    particle_indices = jnp.arange(n_particles)

    @jit
    def get_y_r_vector(r_coords):
        out = []
        for spherical_harmonics in spherical_harmonics_list:
            y_r_matrix = jnp.array([[f(r) for f in spherical_harmonics] for r in r_coords],
                                   dtype=jnp.float64)  # [n_particles, n_func]
            y_r_vector = y_r_matrix[particle_indices, function_permutations]  # [n_permutations, n_functions]
            y_r_vector = vmap(jnp.prod)(y_r_vector)
            out.append(y_r_vector)
        return jnp.array(out, dtype=jnp.float64)

    def wave_function(parameters, r_coords):
        y_r_vector = get_y_r_vector(r_coords)
        y_r_vector = coefficients * y_r_vector
        y_r_vector = y_r_vector.sum(axis=0)
        new_values = y_r_vector * spin_isospin_wave_function[iso_indices, spin_indices]
        wave_function = index_update(spin_isospin_wave_function, index[iso_indices, spin_indices], new_values)
        return wave_function

    return key, wave_function, params
