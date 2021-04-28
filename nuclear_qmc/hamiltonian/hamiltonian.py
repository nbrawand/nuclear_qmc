import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from functools import partial
from nuclear_qmc.constants.constants import H_BAR, NUCLEON_MASS, H_BAR_SQRD_OVER_2_M


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
    d2_psi = jax.hessian(wave_function.density, argnums=0)(r_coords)
    dim = r_coords.shape[-1] * r_coords.shape[-2]
    d2_psi = d2_psi.reshape(dim, dim)
    d2_psi = d2_psi / psi_density_at_r
    ke = - H_BAR_SQRD_OVER_2_M * jnp.trace(d2_psi)
    return ke


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
    psi_density_at_r = wave_function.density(r_coords)
    kinetic_energy_value = kinetic_energy(wave_function, r_coords, psi_density_at_r)
    potential_energy_value = potential_energy()
    return kinetic_energy_value + potential_energy_value

