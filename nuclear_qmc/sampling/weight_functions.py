import jax.numpy as jnp
from nuclear_qmc.wave_function.wave_function import WaveFunction




def wave_function_prefactor_weight(wave_function: WaveFunction, r_coords):
    return wave_function.psi_prefactor(r_coords, wave_function.params)
