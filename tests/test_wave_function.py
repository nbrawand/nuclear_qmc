from nuclear_qmc.wave_function.wave_function import WaveFunction
import jax.numpy as jnp


class TestWaveFunction:
    WFC = WaveFunction(1, 1, include_isospin=True)

    def test_sigma(self):
        computed = self.WFC.sigma(1.0, 1.0)

    def test_tau(self):
        assert True

    def test_kinetic_energy(self):
        r_coords = jnp.array([[0., 1., 2.], [0., 0., 0.]])
        computed = self.WFC.kinetic_energy(r_coords)
