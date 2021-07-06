from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from tests.angular_momentum.get_total_angular_momentum import get_L_sqrd, get_particle_L_sqrd, get_expected_value, \
    auto_diff_hessian_theta, get_particle_L, auto_diff_theta, get_Li_Lj, rotate_psi, L_sqrd_psi, L_sqrd_psi_axis, \
    L_psi_axis, L_sqrd_psi_total
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
