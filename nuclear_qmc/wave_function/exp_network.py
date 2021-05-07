import jax.numpy as jnp
from jax import jit
from functools import partial
from nuclear_qmc.wave_function.wave_function import WaveFunction


class ExpWaveFunction(WaveFunction):
    def __init__(self, params):
        self.params = params
        super().__init__(n_protons=1, n_neutrons=1)

    def psi_prefactor(self, params, r_coords):
        rcm = jnp.mean(r_coords, axis=0)
        r = r_coords - rcm[None, :]
        delta_r1 = jnp.linalg.norm(r[0, :]) ** 2
        delta_r2 = jnp.linalg.norm(r[1, :]) ** 2
        return jnp.exp(- (delta_r1 + delta_r2) / params[0] ** 2)
