import jax.numpy as jnp
from jax import jit
from functools import partial
from nuclear_qmc.wave_function.wave_function import WaveFunction


class ExpWaveFunction(WaveFunction):
    def __init__(self, params):
        self.params = params
        super().__init__(n_protons=1, n_neutrons=1)

    @partial(jit, static_argnums=(0,))
    def psi_prefactor(self, r_coords, params):
        rcm = jnp.mean(r_coords, axis=0)
        r = r_coords - rcm[None, :]
        delta_r = jnp.linalg.norm(r[0, :] - r[1, :])
        return jnp.exp(-params[0] * delta_r)
