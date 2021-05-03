from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M
from nuclear_qmc.wave_function.wave_function import WaveFunction
import numpy as np
import jax.numpy as jnp


class TestWaveFunction:
    WFC = WaveFunction(1, 1, include_isospin=True)

    def test_sigma(self):
        assert True

    def test_tau(self):
        assert True


