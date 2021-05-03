import jax.numpy as jnp
from nuclear_qmc.wave_function.wave_function import WaveFunction


def v_dot_weight(wave_function: WaveFunction, r_coords):
    psi_r = wave_function.psi(r_coords)
    return jnp.vdot(psi_r, psi_r)


def wave_function_prefactor_weight(wave_function: WaveFunction, r_coords):
    return wave_function.psi_prefactor(r_coords, wave_function.params)
