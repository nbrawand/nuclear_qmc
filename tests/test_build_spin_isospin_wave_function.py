from nuclear_qmc.spin.get_spin_isospin_wave_function import get_number_of_orbital_configurations, \
    build_spin_isospin_system, get_states
import jax.numpy as jnp


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


def test_build_spin_isospin_system_2H():
    n_neutrons = 1
    n_protons = 1
    wfc, indices, permutations = build_spin_isospin_system(n_neutrons, n_protons)
    assert jnp.array_equal(indices[:, 0], jnp.array([0, 0]))
    assert jnp.array_equal(indices[:, 1], jnp.array([0, 1]))
    assert jnp.array_equal(indices[:, 2], jnp.array([0, 0]))
    e = jnp.array([[[1., 0., 0., 0.],
                    [-1., 0., 0., 0.]]])
    assert jnp.array_equal(wfc, e)


def test_get_spin_isospin_wave_function_A3():
    expected = jnp.array(
        [[
            [0., 1., -1., 0., 0., 0., 0., 0.],
            [0., -1., 0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., -1., 0., 0., 0.],
        ]])
    computed, _, _ = build_spin_isospin_system(1, 2)
    assert jnp.array_equal(expected, computed)


def test_build_spin_isospin_system_unique():
    n_neutrons = 3
    n_protons = 3
    _, indices, _ = build_spin_isospin_system(n_neutrons, n_protons)
    org_len = len(indices)
    unq_len = len(set([(*i,) for i in indices]))
    assert unq_len == org_len


def test_get_number_of_orbital_configurations():
    c = get_number_of_orbital_configurations(1)
    assert c == 1
    c = get_number_of_orbital_configurations(2)
    assert c == 1
    c = get_number_of_orbital_configurations(3)
    assert c == 1
    c = get_number_of_orbital_configurations(4)
    assert c == 1
    c = get_number_of_orbital_configurations(5)
    assert c == 5
    c = get_number_of_orbital_configurations(6)
    assert c == 15
