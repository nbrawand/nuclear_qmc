import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from functools import partial
from nuclear_qmc.constants.constants import H_BAR, NUCLEON_MASS, H_BAR_SQRD_OVER_2_M
from nuclear_qmc.wave_function.wave_function import WaveFunction


@partial(jax.jit, static_argnums=(0,))
def v_pair(k, params, r, sz):
    r_ij = r[ip[k], :] - r[jp[k], :]
    r_ij = jnp.sqrt(jnp.sum(r_ij ** 2))
    vr_ij = potential.pionless_2b(r_ij)
    vc_ij = vr_ij[0]
    sz_ij = spin.sp_exch(sz, k, 0)
    vs_ij = vr_ij[2] * (2 * wave_function.psi(params, r, sz_ij) / wave_function.psi(params, r, sz) - 1)
    t_ij = potential.pionless_3b(r_ij)
    return vc_ij, vs_ij, t_ij


"""
@partial(jax.jit, static_argnums=(0,))
def potential_energy(wave_function, r_coords):
    v_ij = jnp.zeros(6)
    gr3b = jnp.zeros(npart)
    V_ijk = 0
    k = jnp.arange(npair)
    vc_ij, vs_ij, t_ij = vmap(v_pair, in_axes=(0, None, None, None))(k, params, r, sz)
    v_ij = index_add(v_ij, 0, jnp.sum(vc_ij[:]))
    v_ij = index_add(v_ij, 2, jnp.sum(vs_ij[:]))

    if (npart > 2):
        for k in range(npair):
            gr3b = index_add(gr3b, index[ip[k]], t_ij[k])
            gr3b = index_add(gr3b, index[jp[k]], t_ij[k])
        V_ijk = 0.5 * jnp.sum(gr3b ** 2) - jnp.sum(t_ij ** 2)
    pe = v_ij[0] + v_ij[2] + V_ijk
    return pe
"""


def potential_energy(wave_function: WaveFunction, r_coords):
    """Potential from arXiv:2007.14282v2 [nucl-th] 13 Apr 2021"""

    C_1 = -487.6128
    ultraviolet_cutoff = 4
    r_ij_sqrd = get_r_ij_sqrd(r_coords, wave_function.particle_pairs)
    exp_neg_r_lambda_4 = jnp.exp(-r_ij_sqrd * ultraviolet_cutoff ** 2 / 4.)
    first_term_coefficient = C_1 * exp_neg_r_lambda_4.sum()

    C_2 = -17.5515
    second_term_coefficients = C_2 * exp_neg_r_lambda_4

    D_0 = jnp.sqrt(677.79890)
    r_ik_r_ij = get_r_ik_r_ij_cycles(r_coords, wave_function.particle_triplets)
    third_term_coefficient = D_0 * jnp.exp(-r_ik_r_ij * ultraviolet_cutoff ** 2 / 4.).sum()

    psi_r = wave_function.psi(r_coords)
    v_psi = (first_term_coefficient + third_term_coefficient) * psi_r
    v_psi += wave_function.sigma(r_coords, second_term_coefficients)
    psi_v_psi = jnp.vdot(psi_r, v_psi)
    return psi_v_psi


@partial(jax.jit, static_argnums=(0,))
def kinetic_energy(wave_function, r_coords, psi_density_at_r):
    """

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]
    psi_density_at_r: float

    Returns
    -------
    float

    """
    d2_psi = jax.hessian(wave_function.weight, argnums=0)(r_coords)
    dim = r_coords.shape[-1] * r_coords.shape[-2]
    d2_psi = d2_psi.reshape(dim, dim)
    d2_psi = d2_psi / psi_density_at_r
    ke = - H_BAR_SQRD_OVER_2_M * jnp.trace(d2_psi)
    return ke


@jit
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


@jit
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


@jit
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


@partial(jax.jit, static_argnums=(0,))
def get_local_energy(wave_function, r_coords):
    """

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]

    Returns
    -------
    float

    """
    psi_density_at_r = wave_function.weight(r_coords)
    kinetic_energy_value = kinetic_energy(wave_function, r_coords, psi_density_at_r)
    potential_energy_value = potential_energy()
    return kinetic_energy_value + potential_energy_value
