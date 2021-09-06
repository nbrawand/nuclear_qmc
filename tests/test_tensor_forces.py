from nuclear_qmc.operators.tensor_forces import get_flip_indices, invert_bit, get_bit, \
    get_sigma_operator_prefactors, get_sigma_x_i, get_sigma_y_i, get_sigma_z_i, normalize_r_ij, get_sij_psi_r
from jax import vmap
import jax.numpy as jnp


def test_invert_bit():
    ints = jnp.arange(4)
    computed = vmap(invert_bit, in_axes=(0, None))(ints, 0)
    expected = jnp.array([1, 0, 3, 2])
    assert jnp.array_equal(computed, expected)


def test_get_flip_indices():
    mass_number = 2
    computed = get_flip_indices(mass_number)
    expected = jnp.array([
        [1, 0, 3, 2],
        [2, 3, 0, 1]
    ])
    assert jnp.array_equal(computed, expected)


def test_get_bit():
    computed = get_bit(2, 0)
    assert computed == 0
    computed = get_bit(2, 1)
    assert computed == 1


def test_get_get_sigma_operator_prefactors():
    mass_number = 2
    x, y, z = get_sigma_operator_prefactors(mass_number)
    assert x == 1.0
    expected_y = jnp.array([
        [-1.j, 1.j, -1.j, 1.j],
        [-1.j, -1.j, 1.j, 1.j],
    ])
    assert jnp.array_equal(y, expected_y)
    expected_z = jnp.array([
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ])
    assert jnp.array_equal(z, expected_z)


def test_get_sigma_i_psi_r():
    mass_number = 2
    _, y_prefactors, z_prefactors = get_sigma_operator_prefactors(mass_number)
    flipped_indices = get_flip_indices(mass_number)
    psi_r = jnp.arange(8).reshape(2, 4)
    ith_particle = 0
    sigma_x = get_sigma_x_i(flipped_indices, ith_particle, psi_r)
    expected_x = jnp.array([
        [1, 0, 3, 2],
        [5, 4, 7, 6]
    ])
    assert jnp.array_equal(sigma_x, expected_x)

    sigma_y = get_sigma_y_i(flipped_indices, y_prefactors, ith_particle, psi_r)
    expected_y = 1.j * jnp.array([
        [1, -0, 3, -2],
        [5, -4, 7, -6]
    ])
    assert jnp.array_equal(sigma_y, expected_y)

    sigma_z = get_sigma_z_i(z_prefactors, ith_particle, psi_r)
    expected_z = jnp.array([
        [-0, 1, -2, 3],
        [-4, 5, -6, 7]
    ])
    assert jnp.array_equal(sigma_z, expected_z)


def test_normalize_r_ij():
    r_ij = jnp.array([
        [0, 1, 1],
        [1, 0, 0],
    ])
    computed = normalize_r_ij(r_ij)
    expected = jnp.array([
        [0, 1 / jnp.sqrt(2), 1 / jnp.sqrt(2)],
        [1, 0, 0],
    ])
    assert jnp.array_equal(computed, expected)


def test_get_sij_psi_r():
    mass_number = 2
    r_ij = jnp.array(
        [1, 0, 0]
    )
    psi_r = jnp.array([
        [1., 1, 1, 0],
        [0., 0, 0, 0]
    ])
    particle_i = 0
    particle_j = 1
    flipped_indices = get_flip_indices(mass_number)
    _, y_prefactors, z_prefactors = get_sigma_operator_prefactors(mass_number)
    sigma_ij = 0.0
    computed = get_sij_psi_r(r_ij, psi_r, particle_i, particle_j, flipped_indices, y_prefactors, z_prefactors, sigma_ij)
    expected = jnp.array([[3. + 0.j, 0. + 0.j, 0. + 0.j, 3. + 0.j],
                          [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]])
    assert jnp.array_equal(computed, expected)

    r_ij = jnp.array([0, 1, 0])
    computed = get_sij_psi_r(r_ij, psi_r, particle_i, particle_j, flipped_indices, y_prefactors, z_prefactors, sigma_ij)
    expected = jnp.array([[-3. + 0.j, 0. + 0.j, 0. + 0.j, -3. + 0.j],
                          [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]])
    assert jnp.array_equal(computed, expected)

    r_ij = jnp.array([0, 0, 1])
    computed = get_sij_psi_r(r_ij, psi_r, particle_i, particle_j, flipped_indices, y_prefactors, z_prefactors, sigma_ij)
    expected = jnp.array([[3. + 0.j, -3. + 0.j, -3. + 0.j, 0. + 0.j],
                          [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]])
    assert jnp.array_equal(computed, expected)
