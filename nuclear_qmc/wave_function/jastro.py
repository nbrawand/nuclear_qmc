import jax.numpy as jnp
from nuclear_qmc.utils.get_cyclic_permutations import get_cyclic_permutations
from jax import vmap
from nuclear_qmc.operators.operators import sigma
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.utils.get_dr_ij import get_r_ij
from nuclear_qmc.utils.get_particle_pairs_index import get_particle_pairs_index


def build_2b_jastro(func_2b, particle_pairs):
    def psi_function(in_params, in_r_coords):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_2b_ij = vmap(func_2b, in_axes=(None, 0))(in_params, r_ij)
        psi = jnp.prod(f_2b_ij)
        return psi

    return psi_function


def build_3b_jastro(func_3b, particle_pairs, particle_triplets):
    triplet_cycles_for_all_triplets = vmap(get_cyclic_permutations)(
        particle_triplets)  # dims = [n_particle_triplets, 4, 3]
    pairs_index = get_particle_pairs_index(particle_pairs)

    def f_ij_f_jk(f, triplet):
        i, j, k = triplet
        return f[pairs_index[i, j]] * f[pairs_index[j, k]]

    def one_minus_sum_f_ij_f_jk(f, triplet_cycles):
        terms = vmap(lambda triplet: f_ij_f_jk(f, triplet))(triplet_cycles)
        result = jnp.sum(terms)
        return 1.0 - result

    def psi_function(in_params, in_r_coords):
        r_ij = get_r_ij(in_r_coords, particle_pairs)
        f_3b_ij = vmap(func_3b, in_axes=(None, 0))(in_params, r_ij)
        three_b_factors = vmap(one_minus_sum_f_ij_f_jk, in_axes=(None, 0))(f_3b_ij, triplet_cycles_for_all_triplets)
        psi = jnp.prod(three_b_factors)
        return psi

    return psi_function
