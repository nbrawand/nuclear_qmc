from jax import vmap, hessian
import jax.numpy as jnp
from jax.ops import index_update, index


def rotate_r(r_coords, theta, axis):
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
    rotated_r_coords = jnp.einsum('ij,pj->pi', rotation_matrix, r_coords)
    return rotated_r_coords


def rotate_psi(func_3d, r_coords, theta, axis):
    r_coords_prime = rotate_r(r_coords, -theta, axis)
    func_out_prime = func_3d(r_coords_prime)
    return func_out_prime


def hessian_theta(func, r_coords, axis):
    return hessian(rotate_psi, argnums=(2,))(func, r_coords, 0.0, axis)[0][0]


def get_expected_value(psi, r_coords, o_psi):
    psi_r = psi(r_coords)
    return jnp.vdot(psi_r, o_psi) / jnp.vdot(psi_r, psi_r)


def get_L_sqrd(psi, r_coords):
    axis = jnp.arange(3)
    # L_x^2 = -d^2_{\theta}
    L_sqrd_psi = -1.0 * vmap(hessian_theta, in_axes=(None, None, 0))(psi, r_coords, axis)
    L_sqrd = vmap(get_expected_value, in_axes=(None, None, 0))(psi, r_coords, L_sqrd_psi)
    L_sqrd = L_sqrd.sum(axis=0)
    return L_sqrd
