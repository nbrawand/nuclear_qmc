from nuclear_qmc.operators.spin_orbit import get_orbit_angular_momentum_L_ij, get_total_spin_Sij, spin_orbit_operator
import jax.numpy as jnp

from nuclear_qmc.operators.tensor_forces import get_flip_indices, get_sigma_operator_prefactors


def test_get_orbit_angular_momentum():
    r_coords = jnp.array([
        [1., 0, 0],
        [0., 0, 0]
    ])
    fixed_matrix = jnp.ones(shape=(2, 4))
    r_ij = jnp.array([r_coords[0] - r_coords[1]])
    psi = lambda _p, _r: 2.j * (_r[0, 1] + _r[1, 2]) * fixed_matrix
    psi_params = jnp.array([])
    particle_pairs = jnp.array([[0, 1]])
    computed = get_orbit_angular_momentum_L_ij(psi, psi_params, r_coords, particle_pairs, r_ij)
    expected = jnp.array([0., 1, 1])

    #  for each pair, iso_spin, and spin the value of L should be equal to the expected array
    for c in computed:
        for cc in c:
            for ccc in cc:
                assert jnp.array_equal(expected, ccc)


def test_get_total_spin_S_ij():
    mass_number = 2
    r_coords = jnp.array([
        [0., 0, 0],
        [0., 0, 0]
    ])
    fixed_matrix = jnp.array([
        [1., 0, 0, 0],
        [0., 0, 0, 0]
    ])
    psi = lambda _p, _r: fixed_matrix
    psi_params = jnp.array([])
    particle_pairs = jnp.array([[0, 1]])
    flipped_indices = get_flip_indices(mass_number)
    _, y_prefactors, z_prefactors = get_sigma_operator_prefactors(mass_number)

    computed = get_total_spin_Sij(psi, psi_params, r_coords, flipped_indices, y_prefactors, z_prefactors,
                                  particle_pairs)
    assert jnp.array_equal(computed[0, 0, :, 0], jnp.array([0. + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0. + 0.j]))  # x
    assert jnp.array_equal(computed[0, 0, :, 1], jnp.array([0. + 0.j, 0.0 - 0.5j, 0.0 - 0.5j, 0. + 0.j]))  # y
    assert jnp.array_equal(computed[0, 0, :, 2], jnp.array([-1. + 0.j, 0.0 + 0.0j, 0.0 + 0.0j, 0. + 0.j]))  # z


def test_spin_orbit_operator():
    mass_number = 2
    r_coords = jnp.array([
        [2., 2, 2],
        [1., 1, 1]
    ])
    fixed_matrix = jnp.array([
        [1., 0, 0, 0],
        [0., 0, 0, 0]
    ])
    r_ij = jnp.array([r_coords[0] - r_coords[1]])
    psi = lambda _p, _r: 2.j * (_r[0, 1] + _r[1, 2]) * fixed_matrix
    psi_params = jnp.array([])
    particle_pairs = jnp.array([[0, 1]])
    flipped_indices = get_flip_indices(mass_number)
    _, y_prefactors, z_prefactors = get_sigma_operator_prefactors(mass_number)

    computed = spin_orbit_operator(psi, psi_params, r_coords, r_ij, flipped_indices, y_prefactors, z_prefactors,
                                   particle_pairs)
    computed = computed[0]
    assert computed[0, 0] == 0. - 6.j
    for i in range(2):
        for j in range(1, 4):
            assert computed[i, j] == 0. + 0.j
