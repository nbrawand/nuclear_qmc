from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import Arxiv_2102_02327v1_Potential
import jax.numpy as jnp
from jax import config
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_system_arrays import get_system_arrays
from nuclear_qmc.utils.get_expectation import get_expectation
from nuclear_qmc.constants.constants import H_BAR

config.update("jax_enable_x64", True)

particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices, iso_bin_rep = get_system_arrays(
    1,
    1, also_return_binary_representation=True)

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


def get_pot(kwargs=None):
    if kwargs is None:
        kwargs = {
            'model_string': 'o'
            , 'R3': 1.0
            , 'include_3body': True
            , 'theory_order': 'nlo'
        }
    return Arxiv_2102_02327v1_Potential(particle_pairs, particle_triplets, spin_exchange_indices
                                        , isospin_exchange_indices, jnp.zeros_like(iso_bin_rep), **kwargs)


def test_build_arxiv_2102_02327v1_2H_zero_delta_r():
    potential = Arxiv_2102_02327v1_Potential(particle_pairs, particle_triplets, spin_exchange_indices,
                                             isospin_exchange_indices, jnp.zeros_like(iso_bin_rep),
                                             model_string='o')
    psi_params = jnp.array([1.], dtype=jnp.float64)
    r = jnp.zeros(shape=(2, 3), dtype=jnp.float64)
    spin = jnp.array([
        [+1., 0., 0., 0.],
        [-1., 0., 0., 0.]
    ])
    psi = lambda p, r: spin
    computed = potential(psi, psi_params, r)
    computed = get_expectation(spin, computed)
    c10 = -7.04040080
    r0 = 1.54592984
    expected = H_BAR * c10 / jnp.pi ** (3. / 2.) / r0 ** 3
    expected = jnp.array(expected, dtype=jnp.float64)
    assert jnp.array_equal(computed.round(8), expected.round(8))


def test_build_arxiv_2102_02327v1_2H():
    spin = jnp.array([
        [+1., 0., 0., 0.],
        [-1., 0., 0., 0.]
    ])
    potential = Arxiv_2102_02327v1_Potential(particle_pairs, particle_triplets, spin_exchange_indices,
                                             isospin_exchange_indices, jnp.zeros_like(iso_bin_rep),
                                             model_string='o')
    psi_fac = 2.0
    psi = lambda p, r: psi_fac * spin
    psi_params = jnp.array([1.], dtype=jnp.float64)
    r = jnp.array([
        [0.0, 0, 0],
        [1.5, 0, 0],
    ], dtype=jnp.float64)
    computed = potential(psi, psi_params, r)
    computed = get_expectation(psi_fac * spin, computed)
    c10 = -7.04040080
    r0 = 1.54592984
    expected = H_BAR * c10 / jnp.pi ** (3. / 2.) / r0 ** 3
    expected *= jnp.exp(-(1.5 / r0) ** 2)
    assert jnp.array_equal(computed.round(8), expected.round(8))


def test_C():
    pot = get_pot()
    R = 1
    rij = 1
    computed = pot.C(rij, R)
    expected = jnp.exp(-1) / jnp.pi ** (3 / 2)
    assert computed == expected


def test_C0():
    pot = get_pot()
    rij = 1.0
    computed = pot.C0(rij)
    expected = jnp.exp(-(1 / pot.R0) ** 2) / jnp.pi ** (3 / 2) / pot.R0 ** 3
    assert computed == expected


def test_C1():
    pot = get_pot()
    rij = 1.0
    computed = pot.C1(rij)
    expected = jnp.exp(-(1 / pot.R1) ** 2) / jnp.pi ** (3 / 2) / pot.R1 ** 3
    assert computed == expected


def test_P0():
    pot = get_pot()
    computed = pot.P_0(1)
    assert computed == 0


def test_P1():
    pot = get_pot()
    computed = pot.P_1(1)
    assert computed == 1


def test_C_total():
    pot = get_pot()
    computed = pot.C_total(1, 1)
    expected = pot.C0(1) * pot.P_0(1) + pot.C1(1) * pot.P_1(1)
    assert computed == expected


def test_d1Calpha():
    pot = get_pot()
    r = 1
    R = 1
    computed = pot.d1Calpha(r, R)
    expected = jnp.exp(-r ** 2 / R ** 2) / R ** 2 * -1 * 2 * r / jnp.pi ** (3 / 2) / R ** 3
    assert computed == expected


def test_d2Calpha():
    pot = get_pot()
    r = 1
    R = 1
    computed = pot.d2Calpha(r, R)
    expected = -2 * jnp.exp(-r ** 2 / R ** 2) * (R ** 2 - 2 * r ** 2) / R ** 4 / jnp.pi ** (3 / 2) / R ** 3
    assert computed == expected


"""
def test_pot():
    pot = get_pot()
    psi = lambda p, _r: jnp.array([
        [1., 0, 0, 0],
        [-1., 0, 0, 0]
    ])
    params = jnp.array([])
    r = jnp.array([
        [0, 0, 0]
        , [1., 0, 0]
    ])
    computed = pot(psi, params, r)
    expected = 0.1
    assert computed == expected
"""
