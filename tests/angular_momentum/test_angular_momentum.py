from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from tests.angular_momentum.get_total_angular_momentum import get_L_sqrd, get_particle_L_sqrd, get_expected_value, \
    auto_diff_hessian_theta, get_particle_L, auto_diff_theta, get_Li_Lj, rotate_psi, L_sqrd_psi, L_sqrd_psi_axis, \
    L_psi_axis
from nuclear_qmc.sampling.sample import sample
import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from nuclear_qmc.wave_function.spherical_harmonics import Y11, Y10, Y1m1, get_phi


def test_L_sqrd_psi_axis():
    r_coords = jnp.array([np.random.random(size=(3,))])
    psi = lambda r: jnp.cos(get_phi(r[0]))  # -> partial_theta_z^2 cos(theta) = - cos(theta)
    computed = L_sqrd_psi_axis(psi, r_coords, auto_diff_hessian_theta, 0, 2) / psi(r_coords)
    expected = jnp.array(1.)
    assert jnp.array_equal(computed.round(4), expected)

    psi = lambda r: Y10(r[0])
    computed = L_sqrd_psi_axis(psi, r_coords, auto_diff_hessian_theta, 0, 2) / psi(r_coords)
    expected = jnp.array(0.)
    assert jnp.array_equal(computed, expected)

    psi = lambda r: Y11(r[0])
    computed = L_sqrd_psi_axis(psi, r_coords, auto_diff_hessian_theta, 0, 2) / psi(r_coords)
    expected = jnp.array(1.)
    assert jnp.array_equal(computed.round(4), expected)

    psi = lambda r: Y1m1(r[0])
    computed = L_sqrd_psi_axis(psi, r_coords, auto_diff_hessian_theta, 0, 2) / psi(r_coords)
    expected = jnp.array(1.)
    assert jnp.array_equal(computed.round(4), expected)


def test_L_psi_axis():
    r_coords = jnp.array([np.random.random(size=(3,))])
    psi = lambda r: jnp.cos(get_phi(r[0]))  # -> partial_theta_z cos(theta) = - sin(theta)
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2) / jnp.sin(get_phi(r_coords[0]))
    expected = jnp.array(1.j)
    assert jnp.array_equal(computed.round(4), expected)

    psi = lambda r: Y10(r[0])
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2)
    expected = jnp.array(0.j)
    assert jnp.array_equal(computed.round(4), expected)

    psi = lambda r: Y11(r[0])
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2) / psi(r_coords)
    expected = jnp.array(1.j)
    assert jnp.array_equal(computed.round(4), expected)

    # psi = lambda r: Y1m1(r[0])


def test_angular_momentum_5():
    r_coords = jnp.array([[1., 1., 1.0]])
    psi = lambda r: Y1m1(r[0])
    computed = get_L_sqrd(psi, r_coords, use_auto_diff=True)
    expected = jnp.array(2.)  # L^2 |psi> = l ( l + 1 ) |psi> thus  2 = l^2+l gives l=-2 or l=1 and we take the positive
    assert jnp.array_equal(computed.round(2), expected)


def test_L():
    r_coords = jnp.array([[1.321, 1.123, 3.123]])

    psi = lambda r: Y10(r[0])
    computed = get_particle_L(psi, r_coords, 0, auto_diff_theta, 2)
    expected = jnp.array(0 + 0.j)
    assert jnp.array_equal(computed, expected)

    psi = lambda r: Y1m1(r[0])
    computed = get_particle_L(psi, r_coords, 0, auto_diff_theta, 2) / psi(r_coords)
    expected = jnp.array(0 - 1.j)
    assert jnp.array_equal(computed, expected)

    psi = lambda r: Y11(r[0])
    computed = get_particle_L(psi, r_coords, 0, auto_diff_theta, 2) / psi(r_coords)
    expected = jnp.array(0 + 1.j)
    assert jnp.array_equal(computed, expected)


def test_Li_Lj():
    r_coords = jnp.array([[1., 1., 1.0], [1., 1.5, 2.]])
    psi = lambda r: Y11(r[0]) * Y11(r[1])
    computed = get_particle_L(psi, r_coords, 1, auto_diff_theta, 2) / psi(
        r_coords) ** 2  # * get_particle_L(psi, r_coords, 1, auto_diff_theta, 2) / psi(r_coords)
    print(computed)
    computed = get_Li_Lj(psi, r_coords, 0, 1, auto_diff_theta, 2) / psi(r_coords)
    print(computed)
    computed = get_Li_Lj(psi, r_coords, 0, 1, auto_diff_theta, 1) / psi(r_coords)
    print(computed)
    computed = get_Li_Lj(psi, r_coords, 0, 1, auto_diff_theta, 0) / psi(r_coords)
    print(computed)


def test_angular_momentum_6():
    r_coords = jnp.array([[1., 1., 1.0], [1.35, 1.8, 1.9]])
    c = jnp.sqrt(1. / 3.)
    psi = lambda r: c * Y11(r[0]) * Y1m1(r[1]) + c * Y11(r[1]) * Y1m1(r[0]) - c * Y10(r[1]) * Y10(r[0])
    computed = get_L_sqrd(psi, r_coords, use_auto_diff=True)
    expected = jnp.array(2.)  # L^2 |psi> = l ( l + 1 ) |psi> thus  2 = l^2+l gives l=-2 or l=1 and we take the positive
    print(computed, expected)
    # assert jnp.array_equal(computed.round(2), expected)

#
#
# def test_get_angular_momentum_Li():
#     # build wfc
#     n_proton = 3
#     n_neutron = 3
#     key = jax.random.PRNGKey(0)
#     key, orbital_psi, orbital_psi_params = build_wave_function(key
#                                                                , n_neutron
#                                                                , n_proton
#                                                                , 1
#                                                                , 1)
#     psi_vector = 1.0
#     key = jax.random.PRNGKey(0)
#     key, samples = sample(orbital_psi
#                           , orbital_psi_params
#                           , psi_vector
#                           , n_steps=2
#                           , walker_step_size=0.2
#                           , n_walkers=1000
#                           , n_particles=n_neutron + n_proton
#                           , n_dimensions=3
#                           , n_equilibrium_steps=2
#                           , n_void_steps=300
#                           , key=key
#                           , initial_walker_standard_deviation=1.0
#                           )
#     samples = samples.reshape(-1, n_neutron + n_proton, 3)
#     func_3d = lambda r: orbital_psi(orbital_psi_params, r)
#     computed = vmap(get_total_angular_momentum_z, in_axes=(None, 0))(func_3d, samples)
#     computed = computed.mean().round(2)
#     expected = 0.0
#     # assert computed == expected
#
#     computed = vmap(get_total_angular_momentum, in_axes=(None, 0))(func_3d, samples)
#     computed = computed.mean().round(2)
#     expected = 0.0
#     assert computed == expected
