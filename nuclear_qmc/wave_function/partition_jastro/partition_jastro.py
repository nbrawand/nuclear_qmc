import numpy as np
import jax.numpy as jnp
from jax import jit
from nuclear_qmc.wave_function.neural_network_jastro_builder.get_deepset_jastro import get_deepset_jastro
from itertools import product


def get_partition(orbitals):
    distinct = np.unique(orbitals)
    return [np.where(orbitals == d)[0] for d in distinct]


def get_partitions(permutations, unique=True):
    if unique:
        permutations = np.unique(permutations, axis=0)
    partitions = [get_partition(p) for p in permutations]
    return partitions


def get_partition_indices(permutations):
    permutations = np.array([''.join(p) for p in permutations])

    unique_permutations = np.unique(permutations, axis=0)
    unique_permutations = np.array([''.join(p) for p in unique_permutations])

    indices = np.array([np.where(p == unique_permutations)[0] for p in permutations]).reshape(-1)
    return indices


def transform_partitions_to_matrix_format(partitions):
    """output jnp.array where columns are: particle_number, partition_number, the_radial_orbital_group
    """
    out = []
    part_i = 0
    for partition in partitions:
        sub_part_j = 0
        for sub_partition in partition:
            for elm in sub_partition:
                out.append([elm, part_i, sub_part_j])
            sub_part_j += 1
        part_i += 1
    return jnp.array(out)


def get_partition_pair_indices(partitions):
    """
    The output has rows equal to the total number of pairs in each partition combinations. The columns have the following
    meaning: [particle_1_index, particle_2_index, partition_index, sub_partition_index]

    a sub_partition_index can indicate: S-S, P-P, S-P.
    """
    out = []
    partition_indx = 0
    for partition in partitions:
        sub_partition_len = len(partition)
        sub_partition_indx = 0
        for i in range(sub_partition_len):
            for j in range(i, sub_partition_len):
                pi = partition[i]
                pj = partition[j]
                combos = list(product(pi, pj))
                combos = [tuple(sorted([c[0], c[1]])) for c in combos if c[0] != c[1]]
                combos = set(combos)
                for c in combos:
                    out.append(list(c) + [partition_indx, sub_partition_indx])
                sub_partition_indx += 1
        partition_indx += 1

    return jnp.array(out)


def get_partition_r_ijs(r_coords, partition_pair_indices):
    out = r_coords[partition_pair_indices[:, 0]] - r_coords[partition_pair_indices[:, 1]]
    out = jnp.linalg.norm(out, axis=1)
    return out


def make_function_with_param_slice(slice_start, slice_stop, func):
    return lambda _p, _r: func(_p[slice_start:slice_stop], _r)


def get_partition_jastro(key
                         , radial_orbital_permutations
                         , n_dense
                         , n_hidden_layers, latent_shape=6, debug=False):
    partitions = get_partitions(radial_orbital_permutations, unique=True)
    partition_indices = get_partition_indices(radial_orbital_permutations)
    # partition_pair_indices = get_partition_pair_indices(partitions)
    # partitions = transform_partitions_to_matrix_format(partitions)
    # particle_indices = partitions[:, 0]
    # partition_indices = partitions[:, 1]
    # function_indices = partitions[:, 2]
    # unique_function_indices = jnp.unique(function_indices)
    # unique_partition_indices = jnp.unique(partition_indices)

    deepsets = []
    params = jnp.array([])
    deepset_start_stops = []
    for _ in range(len(partitions[0])):
        if not debug:
            key, deepset_func, deepset_params = get_deepset_jastro(key, n_dense, n_hidden_layers, out_shape=1,
                                                                   in_shape=(3,), latent_shape=latent_shape,
                                                                   wrapper_func=jnp.exp)
        else:
            deepset_func, deepset_params = lambda _p, _r: _r.sum(), jnp.array([])

        start = len(params)
        params = jnp.concatenate((params, deepset_params))
        stop = len(params)
        func = make_function_with_param_slice(start, stop, deepset_func)
        deepsets.append(func)

        deepset_start_stops.append([start, stop])

    @jit
    def psi(_params, _r):
        out = []
        for partition in partitions:
            y = 1.0
            for sub_group_indices, func in zip(partition, deepsets):
                x = _r[sub_group_indices]
                y *= func(_params, x)
            out.append(y)
        return jnp.array(out)[partition_indices]

    return key, psi, params
