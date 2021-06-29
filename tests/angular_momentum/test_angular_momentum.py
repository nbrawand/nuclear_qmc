from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from tests.angular_momentum.get_total_angular_momentum import finite_diff_theta

from nuclear_qmc.sampling.sample import sample
from nuclear_qmc.constants.constants import H_BAR
import jax.numpy as jnp
import numpy as np
from tests.angular_momentum.get_total_angular_momentum import get_total_angular_momentum_z, get_rotate_r, rotate_z
import jax

from jax import vmap, jacfwd
from nuclear_qmc.wave_function.spherical_harmonics import Y11, Y10, Y1m1


def get_theta(r):
    theta = jnp.arctan2(r[1], r[0])
    return theta


def get_phi(r):
    r_mag = jnp.linalg.norm(r)
    phi = jnp.arccos(r[2] / r_mag)
    return phi


def get_polar_coords(x_y_z):
    r = jnp.linalg.norm(x_y_z)
    theta = get_theta(x_y_z)
    phi = get_phi(x_y_z)
    return jnp.array([r, theta, phi])


def get_cartesian_coords(r_theta_phi):
    r_mag, theta, phi = r_theta_phi
    x = r_mag * jnp.cos(theta) * jnp.sin(phi)
    y = r_mag * jnp.sin(theta) * jnp.sin(phi)
    z = r_mag * jnp.cos(phi)
    return jnp.array([x, y, z])


def convert_input_to_spherical(func):
    def wrapper(r_theta_phi_coords):
        r_coords = vmap(get_cartesian_coords)(r_theta_phi_coords)
        return func(r_coords)

    return wrapper


def test_finite_diff_theta():
    def cos(r_coords):
        theta = get_theta(r_coords[0])
        return jnp.cos(theta)

    # check cos value
    r_coords = jnp.array([[0., 1., 1.]])
    computed = cos(r_coords)
    expected = 0.0
    assert computed.round(7) == expected

    # check cos value
    r_coords = jnp.array([[1., 0., 1.]])
    computed = cos(r_coords)
    expected = 1.0
    assert computed.round(7) == expected

    # test finite diff partial theta
    r_coords = jnp.array([[1., 0., 1.]])
    computed = finite_diff_theta(cos, r_coords, 0)
    expected = 0.0
    assert computed.round(4) == expected
    r_coords = jnp.array([[0., 1., 1.]])
    computed = finite_diff_theta(cos, r_coords, 0)
    expected = -1.0
    assert computed.round(3) == expected
    r_coords = jnp.array([[1., -1., 1.]])
    computed = finite_diff_theta(cos, r_coords, 0)
    expected = 0.708
    assert computed.round(3) == expected


def test_angular_momentum_yl0():
    r_coords = jnp.array([[1.0, 1.0, 8.324]])
    func = lambda r: Y10(r[0])
    z = finite_diff_theta(func, r_coords, 0)
    assert z == 0.0


def test_angular_momentum_ylm1():
    r_coords = jnp.array([[1.0, 1.0, 8.324]])
    func = lambda r: Y1m1(r[0])
    z = finite_diff_theta(func, r_coords, 0)
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    phi = get_phi(r_coords[0])
    theta = get_theta(r_coords[0])
    assert z.round(3) == (sq34pi * jnp.cos(theta) * jnp.sin(phi)).round(3)


def test_angular_momentum_yl1():
    r_coords = jnp.array([[1.0, 1.0, 8.324]])
    func = lambda r: Y11(r[0])
    z = finite_diff_theta(func, r_coords, 0)
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    phi = get_phi(r_coords[0])
    theta = get_theta(r_coords[0])
    expected = -sq34pi * jnp.sin(theta) * jnp.sin(phi)
    assert z.round(3) == expected.round(3)


def test_angular_momentum_yl1_with_radial():
    r_coords = jnp.array(np.random.random(size=(1, 3)))
    r_func = lambda r: jnp.exp(jnp.linalg.norm(r[0]))
    func = lambda r: r_func(r) * Y11(r[0])
    z = finite_diff_theta(func, r_coords, 0)
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    phi = get_phi(r_coords[0])
    theta = get_theta(r_coords[0])
    expected = -sq34pi * jnp.sin(theta) * jnp.sin(phi) * r_func(r_coords)
    assert z.round(2) == expected.round(2)


def test_get_angular_momentum_he():
    n_proton = 2
    n_neutron = 2
    key = jax.random.PRNGKey(0)
    key, orbital_psi, orbital_psi_params = build_wave_function(key
                                                               , n_neutron
                                                               , n_proton
                                                               , 1
                                                               , 1)
    func = lambda r: orbital_psi(orbital_psi_params, r)
    r_coords = jnp.array(np.random.random(size=(n_proton + n_neutron, 3)))
    computed = get_total_angular_momentum_z(func, r_coords)
    expected = jnp.zeros(shape=n_proton + n_neutron)
    assert jnp.array_equal(computed, expected)


def test_get_angular_momentum_li():
    n_proton = 3
    n_neutron = 3
    key = jax.random.PRNGKey(0)
    key, orbital_psi, orbital_psi_params = build_wave_function(key
                                                               , n_neutron
                                                               , n_proton
                                                               , 1
                                                               , 1)
    func = lambda r: orbital_psi(orbital_psi_params, r)
    r_coords = jnp.array(np.random.random(size=(n_proton + n_neutron, 3)))
    computed = get_total_angular_momentum_z(func, r_coords)
    expected = jnp.zeros(shape=n_proton + n_neutron)
    assert jnp.array_equal(computed.sum(), expected)


def test_get_angular_momentum_Li():
    # build wfc
    n_proton = 3
    n_neutron = 3
    key = jax.random.PRNGKey(0)
    key, orbital_psi, orbital_psi_params = build_wave_function(key
                                                               , n_neutron
                                                               , n_proton
                                                               , 1
                                                               , 1)
    psi_vector = 1.0
    key = jax.random.PRNGKey(0)
    key, samples = sample(orbital_psi
                          , orbital_psi_params
                          , psi_vector
                          , n_steps=2
                          , walker_step_size=0.2
                          , n_walkers=1000
                          , n_particles=n_neutron + n_proton
                          , n_dimensions=3
                          , n_equilibrium_steps=2
                          , n_void_steps=300
                          , key=key
                          , initial_walker_standard_deviation=1.0
                          )
    samples = samples.reshape(-1, n_neutron + n_proton, 3)
    func_3d = lambda r: orbital_psi(orbital_psi_params, r)
    computed = vmap(get_total_angular_momentum_z, in_axes=(None, 0))(func_3d, samples)
    computed = computed.mean().round(2)
    expected = 0.0
    assert computed == expected
