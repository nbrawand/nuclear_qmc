from nuclear_qmc.wave_function.jastro import build_2b_jastro, build_3b_jastro, build_sigma_jastro, build_tau_jastro
import jax.numpy as jnp


def test_build_2b_jastro():
    pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
    func_2b = lambda params, r_ij: r_ij ** 2
    psi = build_2b_jastro(func_2b, pairs)
    r = jnp.array([
        [0, 0, 0]
        , [1, 0, 0]
        , [2, 0, 0]
    ])
    computed = psi(None, r)
    expected = jnp.prod(jnp.array([1, 4, 1]))
    assert jnp.array_equal(computed, expected)


def test_build_3b_jastro():
    pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
    particle_triplets = jnp.array([
        [0, 1, 2]
        , [2, 1, 0]
    ])
    func_3b = lambda params, r_ij: r_ij
    psi = build_3b_jastro(func_3b, pairs, particle_triplets)
    r = jnp.array([
        [0, 0, 0.0]
        , [1, 0, 0]
        , [2, 0, 0]
    ])
    computed = psi(None, r)
    val_1 = 1. - (1 + 2 + 2)  # 1-(u01u12+u20u01+u12u20)
    val_2 = 1. - (1 + 2 + 2)
    expected = val_1 * val_2
    assert jnp.array_equal(computed, expected)


def test_build_sigma_jastro():
    pairs = jnp.array([
        [0, 1],
        [0, 2],
        [1, 2]])
    spin = jnp.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    func = lambda p, rc: 1.
    x_indices = jnp.array([
        [1, 0, 0],
        [0, 1, 1],
        [2, 2, 2]
    ])
    r = jnp.zeros(shape=(3, 3))
    x_spin = jnp.array([
        [3, 0, 3],
        [6, 3, 6]
    ])
    expected = 2 * spin + x_spin
    psi = build_sigma_jastro(func, pairs, x_indices)
    computed = psi(None, r, spin)
    assert jnp.array_equal(expected, computed)


def test_build_tau_jastro():
    pairs = jnp.array([
        [0, 1],
        [0, 2],
        [1, 2]])
    spin = jnp.array([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    func = lambda p, rc: 1.
    x_indices = jnp.array([
        [1, 0, 0],
        [0, 1, 1],
        [2, 2, 2]
    ])
    r = jnp.zeros(shape=(3, 3))
    x_spin = 2 * jnp.array([
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0]
    ]) - spin
    expected = 2 * spin + x_spin
    psi = build_tau_jastro(func, pairs, x_indices)
    computed = psi(None, r, spin)
    assert jnp.array_equal(expected, computed)
