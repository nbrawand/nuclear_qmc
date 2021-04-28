import jax.numpy as jnp

from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
from nuclear_qmc.wave_function.wave_function_base import WaveFunctionBase


class WaveFunctionSingleOrbital(WaveFunctionBase):

    def __init__(self, n_protons, n_neutrons):
        self._n_protons = n_protons
        self._n_neutrons = n_neutrons
        self._n_particles = self._n_protons + self._n_neutrons
        self.parameters = 0.5 * jnp.ones(self._n_particles, dtype=jnp.float64)
        self.spin_isospin = get_spin_isospin_wave_function(self._n_protons, self._n_neutrons)

    def density(self, r_coords):
        r_mag = jnp.linalg.norm(r_coords, axis=-1)
        density = jnp.exp(self.parameters * r_mag)
        density = jnp.prod(density, axis=-1)
        return density
