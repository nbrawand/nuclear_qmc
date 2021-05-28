from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import build_arxiv_2102_02327v1, \
    get_01_and_10_channels
import jax.numpy as jnp
from jax import config
from nuclear_qmc.wave_function.utility import get_wave_function_system
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.constants.constants import H_BAR

config.update("jax_enable_x64", True)

particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(1,
                                                                                                                    1)


def test_get_01_and_10_channels_2H():
    c_01, c_10 = get_01_and_10_channels(spin, spin_exchange_indices, isospin_exchange_indices)
    assert c_01 == 0.0
    assert c_10 == 1.0


"""
def test_get_01_and_10_channels_3H():
    particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
        1,
        2)
    c_01, c_10 = get_01_and_10_channels(spin, spin_exchange_indices, isospin_exchange_indices)
    ? expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.array_equal(expected, c_01)
    ? expected = jnp.array([1.0, 0.0, 0.0])
    assert jnp.array_equal(expected, c_10)
"""


def test_build_arxiv_2102_02327v1_2H_zero_delta_r():
    potential = build_arxiv_2102_02327v1(spin, particle_pairs, spin_exchange_indices, isospin_exchange_indices,
                                         model_string='o')
    psi = lambda p, r: 1.
    psi_params = jnp.array([1.], dtype=jnp.float64)
    r = jnp.zeros(shape=(2, 3), dtype=jnp.float64)
    computed = potential(psi, psi_params, spin, r)
    computed = get_expectation(spin, computed)
    c10 = -7.04040080
    r0 = 1.54592984
    expected = H_BAR * c10 / jnp.pi ** (3. / 2.) / r0 ** 3
    expected = jnp.array(expected, dtype=jnp.float64)
    assert jnp.array_equal(computed.round(8), expected.round(8))


def test_build_arxiv_2102_02327v1_2H():
    potential = build_arxiv_2102_02327v1(spin, particle_pairs, spin_exchange_indices, isospin_exchange_indices,
                                         model_string='o')
    psi_fac = 2.0
    psi = lambda p, r: psi_fac
    psi_params = jnp.array([1.], dtype=jnp.float64)
    r = jnp.array([
        [0.0, 0, 0],
        [1.5, 0, 0],
    ], dtype=jnp.float64)
    computed = potential(psi, psi_params, spin, r)
    computed = get_expectation(psi_fac * spin, computed)
    c10 = -7.04040080
    r0 = 1.54592984
    expected = H_BAR * c10 / jnp.pi ** (3. / 2.) / r0 ** 3
    expected *= jnp.exp(-(1.5 / r0) ** 2)
    assert jnp.array_equal(computed.round(8), expected.round(8))
