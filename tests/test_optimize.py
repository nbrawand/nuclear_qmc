from nuclear_qmc.optimize.optimize import get_new_wave_function_parameters, partial_psi_prefactor_parameters, \
    partial_full_psi_parameters
from nuclear_qmc.operators.operators import kinetic_energy_psi
import jax.numpy as jnp

from nuclear_qmc.wave_function.wave_function import WaveFunction


class TestOptimize:
    def test_get_new_wave_function_parameters(self):
        wfc = WaveFunction(1, 1)
        wfc.params = jnp.array([1., 1., 1.])
        wfc.psi_prefactor = lambda x, p: x[0, 0] * p[0]
        # wfc.psi_vector = lambda x, p, s: jnp.array([[1., 0.], [0., 2.]])
        n_walkers = 10
        r_coords = n_walkers * [[[1., 0., 0.], [0., 0., 0.]]]
        r_coords = jnp.array(r_coords)
        learning_rate = 0.001
        computed = get_new_wave_function_parameters(wfc
                                                    , r_coords
                                                    , learning_rate
                                                    , partial_function=partial_full_psi_parameters
                                                    , kinetic_energy_operator=kinetic_energy_psi)
