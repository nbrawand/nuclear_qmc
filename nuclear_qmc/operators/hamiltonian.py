import jax
from nuclear_qmc.operators.operators import kinetic_energy_psi, sigma
import jax.numpy as jnp
from jax import jit
from functools import partial

# potential enregy coefficients from arXiv:2007.14282v2 [nucl-th] 13 Apr 2021
C_1 = -487.6128
C_2 = -17.5515
D_0 = 677.79890
ultraviolet_cutoff = 4


def potential_energy_psi(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
                         spin_exchange_indices):
    """Potential from arXiv:2007.14282v2 [nucl-th] 13 Apr 2021"""

    r_ij_sqrd = get_r_ij_sqrd(r_coords, particle_pairs)
    exp_neg_r_lambda_4 = jnp.exp(-r_ij_sqrd * ultraviolet_cutoff ** 2 / 4.)
    first_term_coefficient = C_1 * exp_neg_r_lambda_4.sum()

    second_term_coefficients = C_2 * exp_neg_r_lambda_4

    if particle_triplets.shape[0] > 0:
        r_ik_r_ij = get_r_ik_r_ij_cycles(r_coords, particle_triplets)
        third_term_coefficient = D_0 * jnp.exp(-r_ik_r_ij * ultraviolet_cutoff ** 2 / 4.).sum()
    else:
        third_term_coefficient = 0.0

    psi_r = psi(psi_params, r_coords) * psi_vector
    v_psi = (first_term_coefficient + third_term_coefficient) * psi_r
    v_psi += sigma(psi, psi_params, psi_vector, r_coords, spin_exchange_indices, second_term_coefficients)
    return v_psi


def get_r_ij_sqrd(r_coords, particle_pairs):
    """

    Parameters
    ----------
    r_coords: ndarray[n_particles, n_dimensions]
    particle_pairs: ndarray[n_pairs, 2] the index of each particle in r_coords

    Returns
    -------
    ndarray[n_pairs]
        (r_i-r_j)^2 for each combo i<j, j in order of particle_pairs
    """
    r_ij_sqrd = r_coords[particle_pairs[:, 0]] - r_coords[particle_pairs[:, 1]]
    r_ij_sqrd = (r_ij_sqrd ** 2).sum(axis=-1)
    return r_ij_sqrd


def get_r_ik_r_ij_sqrd(r_coords, particle_triplets, i, j, k):
    """
    Parameters
    ----------
    r_coords: ndarray[n_particles, n_dimensions]
    particle_triplets: ndarray[n_particle_triplets, 3] the index of each particle in r_coords

    Returns
    -------
    ndarray[n_triplets]
        (r_i-r_k)^2+(r_i-r_j)^2 in particle_triplets
    """
    r_ik = r_coords[particle_triplets[:, i]] - r_coords[particle_triplets[:, k]]
    r_ij = r_coords[particle_triplets[:, i]] - r_coords[particle_triplets[:, j]]
    r_ik_ij = (r_ik ** 2).sum(axis=-1) + (r_ij ** 2).sum(axis=-1)
    return r_ik_ij


def get_r_ik_r_ij_cycles(r_coords, particle_triplets):
    """

    Parameters
    ----------
    r_coords: ndarray[n_particles, n_dimensions]
    particle_triplets: ndarray[n_particle_triplets, 3] the index of each particle in r_coords

    Returns
    -------
    ndarray[n_triplets]
        cyclic combinations of ijk of terms: (r_i-r_k)^2+(r_i-r_j)^2 in particle_triplets
    """
    cycles = get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 0, 1, 2)
    cycles = jnp.append(cycles, get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 2, 0, 1))
    cycles = jnp.append(cycles, get_r_ik_r_ij_sqrd(r_coords, particle_triplets, 1, 2, 0))
    return cycles


def hamiltonian_psi(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets, spin_exchange_indices):
    """

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]

    Returns
    -------
    float

    """
    ke_psi = kinetic_energy_psi(psi, psi_params, r_coords) * psi_vector
    v_psi = potential_energy_psi(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
                                 spin_exchange_indices)
    h_psi = ke_psi + v_psi
    return h_psi


def get_local_energy(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets, spin_exchange_indices):
    """

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]

    Returns
    -------
    float

    """
    h_psi = hamiltonian_psi(psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
                            spin_exchange_indices)
    psi_r = psi(psi_params, r_coords) * psi_vector
    psi_psi = jnp.real(jnp.vdot(psi_r, psi_r))
    psi_h_psi = jnp.real(jnp.vdot(psi_r, h_psi))
    return psi_h_psi / psi_psi
