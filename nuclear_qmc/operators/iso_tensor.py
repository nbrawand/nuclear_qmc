import jax.numpy as jnp
from jax import vmap


def get_tau_z_i(z_prefac, psi_r):
    return z_prefac[:, None] * psi_r


def get_iso_tensor_T_ij(psi_r, particle_pairs, z_prefactors, tau_ij):
    """
    _, indices = get_raw_isospin_indices(mass_number, proton_number, as_jax_array=True)
    particle_index = jnp.arange(mass_number)
    extracted_bits = vmap(lambda particle: vmap(get_bit, in_axes=(0, None))(indices, particle))(particle_index)
    z_prefactors = make_negative_1_if_spin_down_else_1(extracted_bits)
    """
    tau_z_i_psi_r = vmap(get_tau_z_i, in_axes=(0, None))(z_prefactors, psi_r)
    out = tau_z_i_psi_r[particle_pairs[:, 0]] * tau_z_i_psi_r[particle_pairs[:, 1]]
    out = 3 * out
    out = out - tau_ij
    return out
