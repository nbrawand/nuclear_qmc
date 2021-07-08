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
    r_coords_prime = rotate_r(r_coords, -theta, ith_particle, axis)
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
    return 1.j * div_function(psi, r_coords, ith_particle, axis)


def L_sqrd_psi(psi, r_coords, div_function, ith_particle):
    axis = jnp.arange(3)
    out = vmap(L_sqrd_psi_axis, in_axes=(None, None, None, None, 0))(psi, r_coords, div_function, ith_particle, axis)
    return out.sum(axis=0)


def get_Li_Lj_axis(psi, r_coords, ith_particle, jth_particle, div_func, axis):
    def d_Lj(_r_coords):
        return L_psi_axis(psi, _r_coords, div_func, jth_particle, axis)

    return L_psi_axis(d_Lj, r_coords, div_func, ith_particle, axis)


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
        Lij = 2.0 * Lij.sum(axis=0)
    else:
        Lij = 0
    return L2 + Lij
