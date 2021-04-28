from nuclear_qmc.wave_function.wave_function import WaveFunction


class TestWaveFunction:
    WFC = WaveFunction(1, 1, include_isospin=True)

    def test_sigma(self):
        computed = self.WFC.sigma(1.0)
        print(computed)

    def test_tau(self):
        assert True
