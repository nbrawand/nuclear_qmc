from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function


class TestGetWFC:
    def test_get_spin_isospin_wave_function(self):
        computed = get_spin_isospin_wave_function(1, 1, True)
