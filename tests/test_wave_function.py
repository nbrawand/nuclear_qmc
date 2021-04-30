from nuclear_qmc.wave_function.wave_function import WaveFunction
import jax.numpy as jnp


class TestWaveFunction:
    WFC = WaveFunction(1, 1, include_isospin=True)

    def test_sigma(self):
        assert True

    def test_tau(self):
        assert True

    def test_kinetic_energy(self):
        r_coords = jnp.array([[0., 1., 2.], [0., 0., 0.]])
        computed = self.WFC.kinetic_energy(r_coords)

    def test_tau_sigma(self):
        wfc = jnp.array([[0., 1., 2., 3], [4., 5, 6, 7]])
        xi = jnp.array([[0, 2, 1, 3], [3, 1, 2, 0]]).T
        pairs = jnp.array([0, 1])
        self.WFC._tau_or_sigma(wfc, xi, pairs)
