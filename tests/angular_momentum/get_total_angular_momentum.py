from jax import vmap, hessian
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


def finite_diff_hessian_theta(func, r_coords, ith_particle, axis):
    dtheta = 0.1
    hess = rotate_psi(func, r_coords, dtheta, ith_particle, axis)
    hess += rotate_psi(func, r_coords, -dtheta, ith_particle, axis)
    hess -= 2.0 * rotate_psi(func, r_coords, 0.0, ith_particle, axis)
    hess /= dtheta ** 2
    return hess


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


def get_L_sqrd(psi, r_coords, use_auto_diff=True):
    if use_auto_diff:
        hessian_func = auto_diff_hessian_theta
    else:
        hessian_func = finite_diff_hessian_theta
    particles = jnp.arange(r_coords.shape[0])
    L_sqrd = vmap(get_particle_L_sqrd, in_axes=(None, None, 0, None))(psi, r_coords, particles, hessian_func)
    return L_sqrd.sum()
