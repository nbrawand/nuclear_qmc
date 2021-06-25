from nuclear_qmc.spin.get_spin_isospin_wave_function import build_wave_function, get_states
from jax import random
import jax.numpy as jnp


class TestGetWFC:

    def test_build_wave_function_2H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=1
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2)
        r = jnp.ones(shape=(2, 3))
        computed = psi(params, r)
        expected = jnp.array([[1., 0., 0., 0.],
                              [-1., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)

    def test_build_wave_function_3H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=2
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2)
        r = jnp.ones(shape=(3, 3))
        computed = psi(params, r)
        expected = jnp.array([[0., 0., 1., 0., -1., 0., 0., 0.],
                              [0., -1., 0., 0., 1., 0., 0., 0.],
                              [0., 1., -1., 0., 0., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)


def test_get_states():
    c = get_states(n_protons=1, n_neutrons=1)
    e = ['Sdn', 'Sdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=1)
    e = ['Sdn', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=1, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee


def test_get_states_pshell():
    c = get_states(n_protons=3, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp', 'Sup', 'Pdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=3)
    e = ['Sdn', 'Sun', 'Pdn', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=3, n_neutrons=3)
    e = ['Sdn', 'Sun', 'Pdn', 'Sdp', 'Sup', 'Pdp']
    for cc, ee in zip(c, e):
        assert cc == ee