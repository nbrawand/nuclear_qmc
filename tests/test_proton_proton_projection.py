from nuclear_qmc.operators.hamiltonian.em_interaction import get_proton_proton_projection
import jax.numpy as jnp
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_spin_isospin_indices import get_system_arrays_pairs_triplets_spin_and_isospin


def test_get_proton_proton_projection():
    particle_pairs, _, _, _, isospin_binary_representation = get_system_arrays_pairs_triplets_spin_and_isospin(
        2, 1, also_return_binary_representation=True)
    computed = get_proton_proton_projection(particle_pairs, isospin_binary_representation)
    expected = jnp.array([[[0],
                           [0],
                           [1]],
                          [[0],
                           [1],
                           [0]],
                          [[1],
                           [0],
                           [0]]])
    assert jnp.array_equal(computed, expected)
