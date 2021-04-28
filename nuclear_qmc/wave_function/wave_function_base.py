import jax.numpy as jnp
from abc import abstractmethod


class WaveFunction:

    @abstractmethod
    def density(self, r_coords):
        return jnp.ones(r_coords.shape[0], dtype=jnp.float64)
