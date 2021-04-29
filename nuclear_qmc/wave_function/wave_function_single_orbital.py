import jax.numpy as jnp
from nuclear_qmc.wave_function.wave_function import WaveFunction


class WaveFunctionSingleOrbital(WaveFunction):

    def __init__(self, n_protons, n_neutrons, include_isospin=True):
        super().__init__(n_protons, n_neutrons, include_isospin=include_isospin)
        self.parameters = 0.5 * jnp.ones(n_protons + n_neutrons, dtype=jnp.float64)

    def psi(self, r_coords):
        r_mag = jnp.linalg.norm(r_coords, axis=-1)
        density = jnp.exp(self.parameters * r_mag)
        density = jnp.prod(density, axis=-1)
        return density * self.spin
