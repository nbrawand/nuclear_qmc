from nuclear_qmc.wave_function.partition_jastro.partition_jastro import get_partitions, get_partition, \
    get_partition_indices, get_partition_pair_indices, get_partition_r_ijs, get_partition_jastro, \
    transform_partitions_to_matrix_format
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey


def test_get_partition():
    orbitals = np.array(['R0', 'R0', 'R1'])
    computed = get_partition(orbitals)
    expected = [np.array([0, 1]), np.array([2])]
    for c, e in zip(computed, expected):
        assert np.array_equal(c, e)


def test_get_partitions():
    orbitals = np.array(['R0', 'R0', 'R1'])
    permutations = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ])
    computed = get_partitions(orbitals[permutations])
    expected = [
        [np.array([0, 1]), np.array([2])],
        [np.array([0, 2]), np.array([1])],
        [np.array([1, 2]), np.array([0])],
    ]
    for c, e in zip(computed, expected):
        for cc, ee in zip(c, e):
            assert np.array_equal(cc, ee)


def test_get_partitions_indices():
    orbitals = np.array(['R0', 'R0', 'R1'])
    permutations = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ])
    computed = get_partition_indices(orbitals[permutations])
    expected = np.array([0, 1, 0, 1, 2, 2])
    assert np.array_equal(computed, expected)


def test_get_partition_r_ijs():
    partition_pair_indices = jnp.array([
        [0, 1, 0, 0],
        [0, 2, 0, 1],
        [1, 2, 0, 1],
        [0, 2, 1, 0],
        [0, 1, 1, 1],
        [1, 2, 1, 1],
    ])
    r_coords = jnp.array([
        [0, 0., 0],
        [0, 0., 1],
        [0, 0., 3],
    ])
    computed = get_partition_r_ijs(r_coords, partition_pair_indices)
    expected = jnp.array([1.0, 3.0, 2.0, 3.0, 1.0, 2.0])
    assert jnp.array_equal(expected, computed)


def test_get_partition_pair_indices():
    partitions = [
        [np.array([0, 1]), np.array([2])],
        [np.array([0, 2]), np.array([1])],
    ]
    computed = get_partition_pair_indices(partitions)
    expected = jnp.array([
        [0, 1, 0, 0],
        [0, 2, 0, 1],
        [1, 2, 0, 1],
        [0, 2, 1, 0],
        [0, 1, 1, 1],
        [1, 2, 1, 1],
    ])
    assert jnp.array_equal(computed, expected)


def test_transform_partitions_to_matrix_format():
    partitions = [[np.array([0, 1]), np.array([2])], [np.array([0, 2]), np.array([1])],
                  [np.array([1, 2]), np.array([0])]]
    computed = transform_partitions_to_matrix_format(partitions)
    expected = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 1],
        [0, 1, 0],
        [2, 1, 0],
        [1, 1, 1],
        [1, 2, 0],
        [2, 2, 0],
        [0, 2, 1],
    ])
    assert jnp.array_equal(computed, expected)


def test_get_partition_jastro():
    key = PRNGKey(0)
    orbitals = np.array(['R0', 'R0', 'R1'])
    permutations = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ])
    key, psi, params = get_partition_jastro(key
                                            , orbitals[permutations]
                                            , n_dense=2
                                            , n_hidden_layers=1, debug=True)
    r_coords = jnp.array([
        [0, 0., 0],
        [0, 0., 1],
        [0, 0., 3],
    ])
    expected = jnp.array([3., 3, 3, 3, 0, 0])
    computed = psi(params, r_coords)
    assert jnp.array_equal(expected, computed)
