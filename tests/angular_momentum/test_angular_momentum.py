from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from nuclear_qmc.wave_function.utility import get_wave_function_system
from tests.angular_momentum.get_total_angular_momentum import get_L_sqrd, get_particle_L_sqrd, get_expected_value, \
    auto_diff_hessian_theta, get_particle_L, auto_diff_theta, get_Li_Lj, rotate_psi, L_sqrd_psi, L_sqrd_psi_axis, \
    L_psi_axis, L_sqrd_psi_total, finite_diff_theta, finite_diff_hessian_theta
from nuclear_qmc.sampling.sample import sample
import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from nuclear_qmc.wave_function.spherical_harmonics import Y11, Y10, Y1m1, get_phi


def test_L_sqrd_single_real_harmonic():
    """L^2 R_l_m = l(l+1) R_l_m"""
    r_coords = jnp.array([np.random.random(size=(3,))])
    particle_pairs = jnp.array([])
    for sphere_func in [Y10, Y11, Y1m1]:
        psi = lambda r: sphere_func(r[0])
        computed = L_sqrd_psi_total(psi, r_coords, auto_diff_hessian_theta, lambda x: None, particle_pairs) / psi(
            r_coords)
        expected = jnp.array(2.)
        assert jnp.array_equal(computed.round(4), expected)


def test_L_z_R11():
    """L_z R_1_1 = i R_1_-1"""
    r_coords = jnp.array([np.random.random(size=(3,))])
    psi = lambda r: Y11(r[0])
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2) / Y1m1(r_coords[0]) / 1.j
    expected = jnp.array(1.0 + 0.j)
    assert jnp.array_equal(computed.round(4), expected)


def test_L_z_R1m1():
    """L_z R_1_-1 = -i R_1_1"""
    r_coords = jnp.array([np.random.random(size=(3,))])
    psi = lambda r: Y1m1(r[0])
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2) / Y11(r_coords[0]) / -1.j
    expected = jnp.array(1.0 + 0.j)
    assert jnp.array_equal(computed.round(4), expected)


def test_L_z_R10():
    """L_z R_1_0 = 0 R_1_0"""
    r_coords = jnp.array([np.random.random(size=(3,))])
    psi = lambda r: Y10(r[0])
    computed = L_psi_axis(psi, r_coords, auto_diff_theta, 0, 2)
    expected = jnp.array(0.0 + 0.j)
    assert jnp.array_equal(computed.round(4), expected)


def test_L_sqrd_2_real_harmonics():
    """-Y11Y11-Y10Y10-Y1m1Y1m1 = 0.0"""
    r_coords = jnp.array(np.random.random(size=(2, 3)))
    particle_pairs = jnp.array([[0, 1]])
    c = jnp.sqrt(1. / 3.)
    psi = lambda r: -c * Y11(r[0]) * Y11(r[1]) - c * Y1m1(r[0]) * Y1m1(r[1]) - c * Y10(r[0]) * Y10(r[1])
    computed = L_sqrd_psi_total(psi, r_coords, finite_diff_hessian_theta, auto_diff_theta, particle_pairs) / psi(
        r_coords)
    expected = jnp.array([0. + 0.j])
    assert jnp.array_equal(computed.round(2), expected)


def test_L_sqrd_2_real_harmonics_2():
    """-Y11Y1m1+Y1m1Y11 = 2.0 |state> """
    r_coords = jnp.array(np.random.random(size=(2, 3)))
    particle_pairs = jnp.array([[0, 1]])
    psi = lambda r: - Y11(r[0]) * Y1m1(r[1]) + Y1m1(r[0]) * Y11(r[1])
    computed = L_sqrd_psi_total(psi, r_coords, finite_diff_hessian_theta, auto_diff_theta, particle_pairs) / psi(
        r_coords)
    expected = jnp.array([2. + 0.j])
    assert jnp.array_equal(computed.round(2), expected)


def test_L_sqrd_of_orbital_wfc():
    key = jax.random.PRNGKey(0)
    n_neu = 1
    n_pro = 1
    r_coords = jnp.array(np.random.random(size=(n_neu + n_pro, 3)))
    particle_pairs, _, _, _, _ = get_wave_function_system(
        n_pro, n_neu, also_return_binary_representation=True)
    key, psi, psi_params = build_wave_function(key, n_neu, n_pro, 1, 1)
    psi_r = lambda r: psi(psi_params, r)
    computed = L_sqrd_psi_total(psi_r, r_coords, finite_diff_hessian_theta, auto_diff_theta, particle_pairs)
    expected = jnp.array(np.zeros(shape=(1, 2, 4)), dtype=jnp.complex64)
    assert jnp.array_equal(computed, expected)


def test_L_sqrd_of_orbital_wfc_li():
    """L^2 |Li> = 0.0"""
    key = jax.random.PRNGKey(0)
    n_neu = 3
    n_pro = 3
    r_coords = jnp.array(np.random.random(size=(n_neu + n_pro, 3)))
    particle_pairs, _, _, _, _ = get_wave_function_system(
        n_pro, n_neu, also_return_binary_representation=True)
    key, psi, psi_params = build_wave_function(key, n_neu, n_pro, 1, 1)
    psi_r = lambda r: psi(psi_params, r)
    # have orbital wave function return just the orbitals dont accumulate the wave function
    # computed = L_sqrd_psi_total(psi_r, r_coords, auto_diff_hessian_theta, auto_diff_theta, particle_pairs)
    # computed = computed/psi_r(r_coords)
    # computed_not_zero = computed[computed.round(4) != 0. + 0.j]
    # print(computed_not_zero)
    # assert len(computed_not_zero) == 0
