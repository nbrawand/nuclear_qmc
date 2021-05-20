from nuclear_qmc.wave_function.jastro import build_2b_jastro, build_3b_jastro
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
