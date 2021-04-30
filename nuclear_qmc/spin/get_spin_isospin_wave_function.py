import jax.numpy as jnp
from nuclear_qmc.spin.get_tables import get_number_of_isospin_states, get_number_of_spin_states
from jax.ops import index, index_update


def get_spin_isospin_wave_function(n_protons, n_neutrons, include_isospin=True, dtype=jnp.float64):
    n_particles = n_protons + n_neutrons
    n_spin_states = get_number_of_spin_states(n_particles)
    n_isospin_states = get_number_of_isospin_states(n_particles, n_protons)

    if include_isospin:
        wave_function = jnp.zeros(shape=(n_isospin_states, n_spin_states), dtype=dtype)
    else:
        wave_function = jnp.zeros(shape=n_spin_states, dtype=dtype)

    if n_protons == 1 and n_neutrons == 1:
        wave_function = index_update(wave_function, index[0, -1], 1.0)  # n up, p up
        if include_isospin:
            wave_function = index_update(wave_function, index[1, -1], -1.0)  # p up, n up

    else:
        RuntimeError(f'get_spin_isospin_wave_function {n_protons} and {n_neutrons} not implemented.')

    return wave_function
