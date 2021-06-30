from nuclear_qmc.wave_function.build_wave_function import build_wave_function
from tests.angular_momentum.get_total_angular_momentum import get_L_sqrd, get_particle_L_sqrd
from nuclear_qmc.sampling.sample import sample
import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from nuclear_qmc.wave_function.spherical_harmonics import Y11, Y10, Y1m1


def test_angular_momentum_one_particle():
    r_coords = jnp.array(np.random.random(size=(1, 3)))
    func = lambda r: Y10(r[0])
    z = get_particle_L_sqrd(func, r_coords, 0)
    assert z == 2.0


def test_angular_momentum_two_particle():
    r_coords = jnp.array(np.random.random(size=(2, 3)))
    func = lambda r: Y10(r[0])
    z = get_particle_L_sqrd(func, r_coords, 0)
    assert z.round(7) == 2.0


def test_angular_momentum_two_particle_multiple_states():
    r_coords = jnp.array(np.random.random(size=(2, 3)))
    c = jnp.sqrt(1. / 3.)

    def slater(r, psi1, psi2):
        return psi1(r[0]) * psi2(r[1]) - psi1(r[1]) * psi2(r[0])

    func = lambda r: c * slater(r, Y11, Y1m1) + c * slater(r, Y1m1, Y11) - c * slater(r, Y10, Y10)
    # func = lambda r: slater(r, Y11, Y1m1) - slater(r, Y1m1, Y11)
    z = get_particle_L_sqrd(func, r_coords, 0) + get_L_sqrd(func, r_coords, 1)
    assert z.round(7) == 4.0


def test_total_angular_momentum():
    r_coords = jnp.array(np.random.random(size=(2, 3)))
    c = jnp.sqrt(1. / 3.)

    def slater(r, psi1, psi2):
        return psi1(r[0]) * psi2(r[1]) - psi1(r[1]) * psi2(r[0])

    func = lambda r: slater(r, Y11, Y1m1)
    # func = lambda r: slater(r, Y11, Y1m1) - slater(r, Y1m1, Y11)
    z = get_L_sqrd(func, r_coords)
    assert z.round(3) == 4.0


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
    computed = get_L_sqrd(func, r_coords)
    print(sum(computed))
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
