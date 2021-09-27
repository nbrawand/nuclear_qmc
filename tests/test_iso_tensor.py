from nuclear_qmc.operators.iso_tensor import get_iso_tensor_T_ij
from nuclear_qmc.operators.tensor_forces import make_negative_1_if_spin_down_else_1, get_bit
from nuclear_qmc.wave_function.get_spin_isospin_indices.get_system_arrays import get_raw_isospin_indices
import jax.numpy as jnp
from jax import vmap


def test_get_iso_tensor_T_ij():
    mass_number = 2
    proton_number = 1
    _, indices = get_raw_isospin_indices(mass_number, proton_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)
    extracted_bits = vmap(lambda particle: vmap(get_bit, in_axes=(0, None))(indices, particle))(particle_index)
    z_prefactors = make_negative_1_if_spin_down_else_1(extracted_bits)
    psi_r = jnp.array([
        [1., 0, 0, 0],
        [0., 0, 0, 0]
    ])
    particle_pairs = jnp.array([
        [0, 1]
    ])
    tau_ij = jnp.zeros_like(psi_r)
    computed = get_iso_tensor_T_ij(psi_r, particle_pairs, z_prefactors, tau_ij)
    expected = jnp.array([
        [[-3., 0., 0., 0.],
         [0., 0., 0., 0.]]
    ])
    assert jnp.array_equal(computed, expected)
