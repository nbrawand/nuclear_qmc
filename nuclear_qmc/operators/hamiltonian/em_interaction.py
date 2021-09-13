import jax.numpy as jnp
from nuclear_qmc.constants.constants import H_BAR, ALPHA, NUCLEON_MASS

B = 4.27


def get_proton_proton_projection(particle_pairs, isospin_binary_representation):
    projection = isospin_binary_representation[:, particle_pairs[:, 0], None] * isospin_binary_representation[:,
                                                                                particle_pairs[:, 1], None]
    projection = jnp.swapaxes(projection, 0, 1)  # [pair, isospin state, 1]
    return projection


def Fc(r_ij):
    x = B * r_ij
    x2 = x ** 2
    x3 = x ** 3
    f = 1. - (1 + 11. / 16. * x + 3. / 16. * x2 + 1. / 48. * x3) * jnp.exp(-x)
    return f


def Fdelta(r_ij):
    x = B * r_ij
    B3 = B ** 3
    x2 = x ** 2
    fdelta = B3 * (1. / 16. + 1. / 16. * x + 1. / 48. * x2) * jnp.exp(-x)
    return fdelta


def build_v_coulomb_proton_proton(particle_pairs, isospin_binary_representation):
    """https://arxiv.org/pdf/nucl-th/9408016.pdf eqn. 4"""
    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)

    def v_coulomb_proton_proton(r_ij, psi_r):
        f = Fc(r_ij)
        v_coulomb_ij = H_BAR * ALPHA * f / r_ij
        r_ij_zero_limit = H_BAR * ALPHA * B * (1. - 11. / 16.)
        v_coulomb_ij = jnp.nan_to_num(v_coulomb_ij, nan=r_ij_zero_limit)
        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = v_coulomb_ij * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out

    return v_coulomb_proton_proton


def build_v_two_photon_proton_proton(particle_pairs, isospin_binary_representation):
    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)

    def potential(r_ij, psi_r):
        f = Fc(r_ij)
        v_c2 = (ALPHA * f / r_ij) ** 2
        v_c2 = - H_BAR * v_c2 / NUCLEON_MASS
        r_ij_zero_limit = - H_BAR * (ALPHA * B * (1. - 11. / 16.)) ** 2 / NUCLEON_MASS
        v_c2 = jnp.nan_to_num(v_c2, nan=r_ij_zero_limit)
        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = v_c2 * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out

    return potential


def build_darwin_foldy_potential(particle_pairs, isospin_binary_representation):
    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)

    def potential(r_ij, psi_r):
        f = Fdelta(r_ij)
        NUCLEON_MASS2 = NUCLEON_MASS ** 2
        out = - H_BAR * ALPHA * f / 4. / NUCLEON_MASS2
        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = out * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out

    return potential


def build_vacuum_polarization(particle_pairs, isospin_binary_representation):
    proton_proton_projection = get_proton_proton_projection(particle_pairs, isospin_binary_representation)

    def potential(r_ij, psi_r):
        """"
        ALPHA2 = ALPHA ** 2
        prefac = 2. * H_BAR * ALPHA2 / 3. / jnp.pi
        out = prefac * Fc(r_ij) / r_ij
        r_ij_zero_limit = prefac * B * (1. - 11. / 16.)
        out = jnp.nan_to_num(out, nan=r_ij_zero_limit)

        psi_r_ij = proton_proton_projection * psi_r
        psi_r_ij = jnp.moveaxis(psi_r_ij, 0, -1)  # move pair axis to end of array to * with r_ij
        out = out * psi_r_ij
        out = out.sum(axis=-1)  # sum over each particle pair
        return out
        """
        pass

    return potential
