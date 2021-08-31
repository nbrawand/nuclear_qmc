from nuclear_qmc.operators.operators import sigma
import jax.numpy as jnp

from nuclear_qmc.utils.get_dr_ij import get_r_ij_sqrd, get_r_ik_r_ij_cycles

C_1 = -487.6128
C_2 = -17.5515
D_0 = 677.79890
ultraviolet_cutoff = 4


def build_arxiv_2007_14282v2(particle_pairs, particle_triplets,
                             spin_exchange_indices):
    """Potential from arXiv:2007.14282v2 [nucl-th] 13 Apr 2021

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    particle_pairs: ndarray
        [n_pairs, 2] particle indices for each pair.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.
    spin_exchange_indices:
        2D array containing the indices after applying :math:`\\sigma_{ij}` psi.

    Returns
    -------
    function

    """

    def potential(psi, psi_params, r_coords):
        r_ij_sqrd = get_r_ij_sqrd(r_coords, particle_pairs)
        exp_neg_r_lambda_4 = jnp.exp(-r_ij_sqrd * ultraviolet_cutoff ** 2 / 4.)
        first_term_coefficient = C_1 * exp_neg_r_lambda_4.sum()

        second_term_coefficients = C_2 * exp_neg_r_lambda_4

        if particle_triplets.shape[0] > 0:
            r_ik_r_ij = get_r_ik_r_ij_cycles(r_coords, particle_triplets)
            third_term_coefficient = D_0 * jnp.exp(-r_ik_r_ij * ultraviolet_cutoff ** 2 / 4.).sum()
        else:
            third_term_coefficient = 0.0

        psi_r = psi(psi_params, r_coords)
        v_psi = (first_term_coefficient + third_term_coefficient) * psi_r
        v_psi += sigma(psi, psi_params, r_coords, spin_exchange_indices, second_term_coefficients)
        return v_psi

    return potential
