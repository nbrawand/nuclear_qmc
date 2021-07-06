from jax import vmap, hessian, jacfwd
from itertools import combinations as get_combinations
import jax.numpy as jnp
from jax.ops import index_update, index


def rotate_r(r_coords, theta, ith_particle, axis):
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    rotation_matrix = jnp.array([
        [[1., 0., 0.],
         [0., cos, -sin],
         [0., sin, cos]],  # x
        [[cos, 0., sin],
         [0., 1., 0.],
         [-sin, 0., cos]],  # y
        [[cos, -sin, 0.],
         [sin, cos, 0.],
         [0., 0., 1.]]  # z
    ])[axis]
    rotated_ith_coords_prime = jnp.einsum('ji,i', rotation_matrix, r_coords[ith_particle])
    r_coords = index_update(r_coords, index[ith_particle], rotated_ith_coords_prime)
    return r_coords


def rotate_psi(func_3d, r_coords, theta, ith_particle, axis):
    r_coords_prime = rotate_r(r_coords, theta, ith_particle, axis)
    func_out_prime = func_3d(r_coords_prime)
    return func_out_prime


def auto_diff_hessian_theta(func, r_coords, ith_particle, axis):
    return hessian(rotate_psi, argnums=(2,))(func, r_coords, 0.0, ith_particle, axis)[0][0]


def auto_diff_theta(func, r_coords, ith_particle, axis):
    return jacfwd(rotate_psi, argnums=(2,))(func, r_coords, 0.0, ith_particle, axis)[0]


def finite_diff_hessian_theta(func, r_coords, ith_particle, axis):
    dtheta = 0.1
    hess = rotate_psi(func, r_coords, dtheta, ith_particle, axis)
    hess += rotate_psi(func, r_coords, -dtheta, ith_particle, axis)
    hess -= 2.0 * rotate_psi(func, r_coords, 0.0, ith_particle, axis)
    hess /= dtheta ** 2
    return hess


def finite_diff_theta(func, r_coords, ith_particle, axis):
    dtheta = 0.1
    hess = rotate_psi(func, r_coords, dtheta, ith_particle, axis)
    hess -= rotate_psi(func, r_coords, -dtheta, ith_particle, axis)
    hess /= 2.0 * dtheta
    return hess


def L_sqrd_psi_axis(psi, r_coords, div_function, ith_particle, axis):
    return - div_function(psi, r_coords, ith_particle, axis)


def L_psi_axis(psi, r_coords, div_function, ith_particle, axis):
    return - 1.j * div_function(psi, r_coords, ith_particle, axis)


def L_sqrd_psi(psi, r_coords, div_function, ith_particle):
    axis = jnp.arange(3)
    out = vmap(L_sqrd_psi_axis, in_axes=(None, None, None, None, 0))(psi, r_coords, div_function, ith_particle, axis)
    return out.sum(axis=0)


def get_Li_Lj_axis(psi, r_coords, ith_particle, jth_particle, div_func, axis):
    def d_Lj(_r_coords):
        return get_particle_L(psi, _r_coords, jth_particle, div_func, axis)

    return get_particle_L(d_Lj, r_coords, ith_particle, div_func, axis)


def get_Li_Lj(psi, r_coords, ith_particle, jth_particle, div_func):
    axis = jnp.arange(3)
    out = vmap(get_Li_Lj_axis, in_axes=(None, None, None, None, None, 0))(psi, r_coords, ith_particle, jth_particle,
                                                                          div_func, axis)
    return out.sum(axis=0)


def L_sqrd_psi_total(psi, r_coords, div_squared_function, div_function, particle_paris):
    """\sum_i L^2_i + 2 \sum_{i<j} L_i \cdot L_j"""

    # sum L^2 term
    particles = jnp.arange(r_coords.shape[0])
    L2 = vmap(L_sqrd_psi, in_axes=(None, None, None, 0))(psi, r_coords, div_squared_function, particles)
    L2 = L2.sum(axis=0)

    # L_i_j term
    if r_coords.shape[0] > 1:  # only for multiple particles
        ith_particle = particle_paris[:, 0]
        jth_particle = particle_paris[:, 1]
        Lij = vmap(get_Li_Lj, in_axes=(None, None, 0, 0, None))(psi, r_coords, ith_particle, jth_particle, div_function)
        Lij = 2.0 * Lij
    else:
        Lij = 0
    return L2 + Lij


####################


def get_expected_value(psi, r_coords, o_psi):
    psi_r = psi(r_coords)
    return jnp.vdot(psi_r, o_psi) / jnp.vdot(psi_r, psi_r)


def get_particle_L_sqrd(psi, r_coords, ith_particle, hessian_func):
    axis = jnp.arange(3)
    # L_x^2 = -d^2_{\theta}
    L_sqrd_psi = -1.0 * vmap(hessian_func, in_axes=(None, None, None, 0))(psi, r_coords, ith_particle, axis)
    L_sqrd = vmap(get_expected_value, in_axes=(None, None, 0))(psi, r_coords, L_sqrd_psi)
    L_sqrd = L_sqrd.sum(axis=0)
    return L_sqrd


def get_particle_L(psi, r_coords, ith_particle, div_func, axis):
    L_psi = 1.j * div_func(psi, r_coords, ith_particle, axis)
    return L_psi


def get_L_sqrd(psi, r_coords, use_auto_diff=True):
    if use_auto_diff:
        hessian_func = auto_diff_hessian_theta
        div_func = auto_diff_theta
    else:
        hessian_func = finite_diff_hessian_theta
        div_func = finite_diff_theta
    particles = jnp.arange(r_coords.shape[0])

    # get L_i^2 terms
    particle_L_sqrd = vmap(get_particle_L_sqrd, in_axes=(None, None, 0, None))(psi, r_coords, particles, hessian_func)
    total_L_sqrd = particle_L_sqrd.sum()

    # get 2 L_i \cdot L_j terms
    # particle_L = vmap(get_particle_L, in_axes=(None, None, 0, None))(psi, r_coords, particles, div_func)
    if r_coords.shape[0] > 1:
        particle_i_js = jnp.array(list(get_combinations(jnp.arange(r_coords.shape[0]), 2)))
        axis = jnp.arange(3)

        def f(pair):
            ff = lambda a: 2 * get_Li_Lj(psi, r_coords, pair[0], pair[1], div_func, a)
            return vmap(ff)(axis).sum(axis=0)

        n_pair_n_wfc = vmap(f)(particle_i_js)
        n_wfc = n_pair_n_wfc.sum(axis=0)
        L_combo_terms = get_expected_value(psi, r_coords, n_wfc)
        total_L_sqrd += L_combo_terms

        # total_L_sqrd += (2.0 * vmap(lambda L_pair: jnp.vdot(L_pair[0], L_pair[1]))(particle_L_combinations)).sum(axis=0)

    return total_L_sqrd
