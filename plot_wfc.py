from jax.config import config
import jax.numpy as jnp
from nuclear_qmc.utils.debug.wave_function_plot import get_wave_function_plot
from main import psi_prefactor, param_file, particle_pairs
import copy

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

psi_params = jnp.load(param_file + '.npy')

plt = get_wave_function_plot(psi_prefactor, psi_params, 2, 3, particle_pairs, 4, step_size=0.05)
plt.savefig('wfc_plot.png')
