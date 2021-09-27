from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import \
    Arxiv_2102_02327v1_Potential as build_arxiv_2102_02327v1
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_system_arrays import get_system_arrays
import jax.numpy as jnp

# def test_next_leading_order_potential_h2():
#     n_proton = 1
#     n_neutron = 1
#     psi_params = jnp.array([])
#     psi = lambda _p, _r: jnp.array([
#         [1., 0, 0, 0],
#         [-1., 0, 0, 0]
#     ])
#     r_coords = jnp.array([
#         [0., 0, 0],
#         [1., 0, 0]
#     ])
#     particle_pairs, particle_triplets, spin_exchange_indices, isospin_exchange_indices, isospin_binary_representation = get_spin_isospin_indices(
#         n_proton, n_neutron, also_return_binary_representation=True)
#     potential = build_arxiv_2102_02327v1(particle_pairs
#                                          , particle_triplets
#                                          , spin_exchange_indices
#                                          , isospin_exchange_indices
#                                          , isospin_binary_representation
#                                          , model_string='o'
#                                          , R3=1.0
#                                          , include_3body=False
#                                          , theory_order='nlo')
#     computed = potential(psi, psi_params, r_coords)
#     print(computed)
#     print(potential.C01)
