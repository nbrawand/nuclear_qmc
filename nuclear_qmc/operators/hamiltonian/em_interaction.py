import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR, ALPHA


def get_proton_proton_projection(particle_pairs, isospin_binary_representation):
    projection = isospin_binary_representation[:, particle_pairs[:, 0], None] * isospin_binary_representation[:,
                                                                                particle_pairs[:, 1], None]
    projection = jnp.swapaxes(projection, 0, 1)  # [pair, isospin state, 1]
    return projection


def build_v_coulomb_proton_proton(particle_pairs, isospin_binary_representation):
    """https://arxiv.org/pdf/nucl-th/9408016.pdf eqn. 4"""
    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)
    b = 4.27

    def v_coulomb_proton_proton(r_ij, psi_r):
        x = b * r_ij
        x2 = x ** 2
        x3 = x ** 3
        f = 1. - (1 + 11. / 16. * x + 3. / 16. * x2 + 1. / 48. * x3) * jnp.exp(-x)
        v_coulomb_ij = H_BAR * ALPHA * f / r_ij
        r_ij_zero_limit = H_BAR * ALPHA * b * (1. - 11. / 16.)
        v_coulomb_ij = jnp.nan_to_num(v_coulomb_ij, nan=r_ij_zero_limit)
        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = v_coulomb_ij * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out

    return v_coulomb_proton_proton
