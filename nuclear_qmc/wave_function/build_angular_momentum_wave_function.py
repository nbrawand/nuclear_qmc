from jax import vmap
from jax.ops import index, index_update
import jax.numpy as jnp
from nuclear_qmc.spin.spherical_harmonics import get_spherical_harmonic_functions, get_spherical_harmonic_systems


def get_spherical_harmonics_vector(spherical_harmonics, r_coords, particle_indices, function_permutations):
    y_r_matrix = jnp.array([[f(r) for f in spherical_harmonics] for r in r_coords],
                           dtype=jnp.complex64)  # [n_particles, n_func]
    y_r_vector = y_r_matrix[particle_indices, function_permutations]  # [n_permutations, n_functions]
    y_r_vector = vmap(jnp.prod)(y_r_vector)
    return y_r_vector


def build_angular_momentum_wave_function(n_particles
                                         , function_permutations
                                         , iso_indices
                                         , spin_indices
                                         , L_total
                                         , L_z_total
                                         , L_1
                                         , L_2
                                         , spin_isospin_wave_function
                                         ):
    coefficients, spherical_harmonics_list = get_spherical_harmonic_systems(n_particles
                                                                            , L_total
                                                                            , L_z_total
                                                                            , L_1
                                                                            , L_2)
    particle_indices = jnp.arange(n_particles)

    def wave_function(parameters, r_coords):
        y_r_vector = jnp.array([get_spherical_harmonics_vector(spherical_harmonics
                                                               , r_coords
                                                               , particle_indices
                                                               , function_permutations)
                                for spherical_harmonics in spherical_harmonics_list])
        y_r_vector = coefficients * y_r_vector
        y_r_vector = y_r_vector.sum(axis=0)
        new_values = y_r_vector * spin_isospin_wave_function[iso_indices, spin_indices]
        wave_function = index_update(spin_isospin_wave_function, index[iso_indices, spin_indices], new_values)
        return wave_function

    return wave_function, jnp.array([])
