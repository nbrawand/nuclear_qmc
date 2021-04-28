import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from functools import partial
from nuclear_qmc.constants.constants import H_BAR, NUCLEON_MASS, H_BAR_SQRD_OVER_2_M


def hamiltonian(n_particles, wave_function, potential):
    mass = NUCLEON_MASS
    hbar = H_BAR
    hbar2m = H_BAR_SQRD_OVER_2_M
    nrho = 100
    rho_min = 0.
    rho_max = 5.
    r_rho = np.linspace(rho_min, rho_max, num=nrho + 1)
    v_rho = np.zeros(shape=[nrho])
    for i in range(nrho):
        v_rho[i] = r_rho[i + 1] ** 3 - r_rho[i] ** 3
    v_rho = 4. / 3. * jnp.pi * v_rho


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
def potential_energy(params, r, sz):
    "Returns potential energy"
    v_ij = jnp.zeros(6)
    gr3b = jnp.zeros(npart)
    V_ijk = 0

    #        vc_ij, vs_ij, t_ij = v_pair(0, params, r, sz)

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
def get_kinetic_energy_jf(wave_function, r_coords, psi_density_at_r):
    d_psi = jax.grad(wave_function.density, argnums=0)(r_coords)
    d_psi = d_psi.reshape(-1)
    d_psi = d_psi / psi_density_at_r
    ke_jf = H_BAR_SQRD_OVER_2_M * jnp.sum(d_psi * d_psi)
    return ke_jf


@partial(jax.jit, static_argnums=(0,))
def energy(params, r, sz):
    ke, ke_jf = vmap(kinetic_energy, in_axes=(None, 0, None), out_axes=(0))(params, r, sz)
    pe = vmap(potential_energy, in_axes=(None, 0, None))(params, r, sz)
    energy_jf = ke_jf + pe
    energy = ke + pe
    return energy, energy_jf


def density(x):
    """Computes the expectation value of the single-nucleon density
    """
    jnp.set_printoptions(threshold=np.inf)
    xcm = jnp.mean(x, axis=1)
    x = x - xcm[:, None, :]
    r = jnp.sqrt(jnp.sum(x ** 2, axis=(2)))
    rho = jnp.histogram(r, bins=nrho, range=(rho_min, rho_max))
    rho = rho[0] / v_rho
    return rho


def density_print(rho, error_rho):
    jnp.set_printoptions(precision=8, threshold=np.inf)
    for i in range(nrho):
        print((r_rho[i] + r_rho[i + 1]) / 2., rho[i], error_rho[i])
    return
