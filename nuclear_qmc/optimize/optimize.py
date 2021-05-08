import jax
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from nuclear_qmc.operators.hamiltonian import hamiltonian_psi, get_local_energy
from nuclear_qmc.operators.operators import kinetic_energy_psi


def d_psi_d_params(psi, psi_params, r_coord):
    return jax.grad(psi, argnums=0)(psi_params, r_coord)


def get_new_wave_function_parameters(
        psi
        , psi_params
        , psi_vector
        , r_coords
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
        , learning_rate):
    """

    Parameters
    ----------
    learning_rate: float
    wave_function: WaveFunction
    energy: ndarray[n_walkers]
    r_coords: ndarray[n_walkers, n_particles, n_dimensions]

    Returns
    -------

    """

    d_psi = vmap(d_psi_d_params, in_axes=(None, None, 0))(psi, psi_params, r_coords)
    d_psi = jnp.tensordot(d_psi, psi_vector, axes=0)  # walkers, params, spin-isospin
    h_psi = vmap(hamiltonian_psi, in_axes=(None, None, None, 0, None, None, None))(
        psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets, spin_exchange_indices)
    psi_r = vmap(psi, in_axes=(None, 0))(psi_params, r_coords)
    psi_r = jnp.tensordot(psi_r, psi_vector, axes=0)

    psi_psi = vmap(jnp.vdot, in_axes=(0, 0))(psi_r, psi_r)
    d_psi_h_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(0, 0))(x, y), in_axes=(1, None))(d_psi, h_psi)
    d_psi_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(0, 0))(x, y), in_axes=(1, None))(d_psi, psi_r)
    psi_h_psi = vmap(jnp.vdot, in_axes=(0, 0))(psi_r, h_psi)
    d_energy = 2.0 * (d_psi_h_psi / psi_psi).mean() - 2.0 * (psi_h_psi / psi_psi).mean() * (d_psi_psi / psi_psi).mean()
    return - learning_rate * d_energy
#
#
# """Condition S+Lambda
#     # return wave_function.params -learning_rate * d_energy
#
#     # quantum fisher information
#     def vdot_nested(in_d_psi):
#         return vmap(jnp.vdot, in_axes=(2, None))(d_psi, in_d_psi)
#
#     d_psi_d_psi = vmap(vdot_nested, in_axes=(2,))(d_psi)  # [n_params, n_params]
#     fisher_information = d_psi_d_psi / psi_psi
#     fisher_information -= jnp.tensordot(d_psi_psi, d_psi_psi, axes=0) / psi_psi ** 2
#
#     max_eps = 0.1
#     min_eps = 0.0000001
#     energy_change_min = 1.0
#     out_params = wave_function.params
#     for n in range(4):
#         eps = (max_eps + min_eps) / 2.0
#         small_diag_matrix = eps * jnp.identity(fisher_information.shape[0])
#         fisher_information = small_diag_matrix + fisher_information
#         cho_factor_solution = cho_factor(fisher_information)
#         delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)
#
#         original_local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coords)
#         original_local_energy_mean = original_local_energy.mean()
#         original_local_energy_std = original_local_energy.std()
#
#         wave_function.params += delta_p
#         new_local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coords)
#         new_local_energy_mean = new_local_energy.mean()
#         new_local_energy_std = new_local_energy.std()
#         wave_function.params -= delta_p
#
#         energy_change = new_local_energy_mean - original_local_energy_mean
#         energy_std_change = new_local_energy_std - original_local_energy_std
#         if energy_change < energy_change_min and energy_std_change < 1.:
#             energy_change_min = energy_change
#             out_params = wave_function.params + delta_p
#             max_eps = eps
#         else:
#             min_eps = eps
#
#     return out_params
# """
