from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import build_arxiv_2102_02327v1, \
    get_01_and_10_channels
import jax.numpy as jnp
from nuclear_qmc.wave_function.utility import get_wave_function_system

particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(1,
                                                                                                                    1)


def test_get_01_and_10_channels():
    c_01, c_10 = get_01_and_10_channels(spin, spin_exchange_indices, isospin_exchange_indices)
    assert c_01 == 0.0
    assert c_10 == 1.0


def test_build_arxiv_2102_02327v1():
    potential = build_arxiv_2102_02327v1(spin, particle_pairs, spin_exchange_indices, isospin_exchange_indices,
                                         model_string='o')
    psi = lambda p, r: 1.
    psi_params = jnp.array([1.])
    r = jnp.zeros(shape=(2, 3))
    computed = potential(psi, psi_params, spin, r)
    print(computed)
