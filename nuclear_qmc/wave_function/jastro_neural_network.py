import jax.numpy as jnp
from jax import jit
from jax import vmap
from nuclear_qmc.operators.operators import sigma
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.utils.get_cyclic_permutations import get_cyclic_permutations
from nuclear_qmc.utils.get_dr_ij import get_r_ij
from nuclear_qmc.wave_function.neural_network import build_nn_wfc
from nuclear_qmc.wave_function.utility import apply_confining_potential


def get_exp_rij(params, r, particle_pairs):
    r1 = r[particle_pairs[:, 0]]
    r2 = r[particle_pairs[:, 1]]
    dr = r1 - r2
    dr = jnp.linalg.norm(dr, axis=1)
    return jnp.exp(-params * dr)


def build_jastro_nn_2_body(key, n_dense, particle_pairs, n_hidden_layers=2):
    key, nn_func, params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        dr_ij = get_r_ij(r_coords, particle_pairs)
        nn_dr_ij = vmap(nn_func, in_axes=(None, 0))(in_params, dr_ij)
        f_c_ij = jnp.exp(nn_dr_ij.sum())
        psi = f_c_ij * apply_confining_potential(r_coords)
        return psi

    return key, psi_function, params


def build_jastro_nn_2_and_3_body(key, n_dense, particle_pairs, particle_triplets, n_hidden_layers=2):
    key, b2_func, b2_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
    key, b3_func, b3_params = build_nn_wfc(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
    n_b2_params = b2_params.shape[0]
    triplet_cyclic_3_indices = vmap(get_cyclic_permutations)(particle_triplets)  # dims = [n_particle_triplets, 4, 3]

    def u_ij_u_jk(indices, params, r_coords):
        i, j, k = indices
        r_ij = jnp.linalg.norm(r_coords[i] - r_coords[j])
        r_jk = jnp.linalg.norm(r_coords[j] - r_coords[k])
        return jnp.exp(b3_func(params, r_ij) + b3_func(params, r_jk))

    def sum_uus(cycle_indices, params, r_coords):
        uus = vmap(u_ij_u_jk, in_axes=(0, None, None))(cycle_indices, params, r_coords)
        return jnp.sum(uus)

    @jit
    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        dr_ij = get_r_ij(r_coords, particle_pairs)

        # b2
        b2_ij = vmap(b2_func, in_axes=(None, 0))(in_params[:n_b2_params], dr_ij)
        b2_ij = jnp.exp(b2_ij.sum())

        # b3
        b3_ij = vmap(lambda c, p, r: 1.0 - sum_uus(c, p, r), in_axes=(0, None, None))(triplet_cyclic_3_indices,
                                                                                      in_params[n_b2_params:], r_coords)
        b3_ij = jnp.prod(b3_ij)

        psi = b2_ij * b3_ij * apply_confining_potential(r_coords)
        return psi

    params = jnp.concatenate((b2_params, b3_params))
    return key, psi_function, params
