from jax import grad, jacfwd, vmap
import jax.numpy as jnp
import numpy as np
from nuclear_qmc.constants.constants import H_BAR
from jax.ops import index_update, index


def get_rotate_r(r_coords, theta, ith_particle, axis):
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


def rotate(func_3d, r_coords, theta, ith_particle, axis):
    r_coords_prime = get_rotate_r(r_coords, theta, ith_particle, axis)
    func_out_prime = func_3d(r_coords_prime)
    return func_out_prime


def get_expected_value(func_3d, r_coords, L_z):
    func_out = func_3d(r_coords)
    total_angular_momentum = jnp.vdot(func_out, L_z) / jnp.vdot(func_out, func_out)
    return total_angular_momentum


def finite_diff_theta(func, r_coords, ith_particle, axis):
    dtheta = 0.0001
    rot_func_r = rotate(func, r_coords, dtheta, ith_particle, axis) - rotate(func, r_coords, -dtheta, ith_particle,
                                                                             axis)
    rot_func_r /= 2. * dtheta
    return rot_func_r


def finite_diff_theta_2nd_order(func, r_coords, ith_particle, axis):
    dtheta = 0.0001
    rot_func_r = rotate(func, r_coords, dtheta, ith_particle, axis)
    rot_func_r += rotate(func, r_coords, -dtheta, ith_particle, axis)
    rot_func_r -= 2 * func(r_coords)
    rot_func_r /= dtheta ** 2
    return rot_func_r


def get_L_sqrd(func, r_coords, ith_particle):
    axis = jnp.arange(3)
    diffs = vmap(finite_diff_theta_2nd_order, in_axes=(None, None, None, 0))(func, r_coords, ith_particle, axis)
    L_sqrd = diffs.sum(axis=0)
    return L_sqrd


def get_total_angular_momentum(func, r_coords):
    particles = jnp.arange(r_coords.shape[0])
    partials = vmap(get_L_sqrd, in_axes=(None, None, 0))(func, r_coords, particles)
    expected_values = vmap(get_expected_value, in_axes=(None, None, 0))(func, r_coords, partials)
    return expected_values


def get_total_angular_momentum_z(func, r_coords):
    particles = jnp.arange(r_coords.shape[0])
    partials = vmap(finite_diff_theta, in_axes=(None, None, 0, None))(func, r_coords, particles, 2)
    expected_values = vmap(get_expected_value, in_axes=(None, None, 0))(func, r_coords, partials)
    return expected_values

    # rotation_angles_rads = vmap(lambda r: jnp.arctan2(r[1], r[0]))(r_coords)
    # particles = jnp.arange(r_coords.shape[0])
    # rotation_func = lambda func, r, angle, particle: jacfwd(rotate_z, argnums=(2,))(func, r, angle, particle)[0]
    # L_z = vmap(rotation_func, in_axes=(None, None, 0, 0))(func_3d, r_coords, rotation_angles_rads, particles)
    # total_angular_momentum = vmap(get_expected_value, in_axes=(None, None, 0))(func_3d, r_coords, L_z)
    # return total_angular_momentum
