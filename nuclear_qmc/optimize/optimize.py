import jax
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from nuclear_qmc.operators.hamiltonian import hamiltonian_psi


def d_psi_d_params(psi, psi_params, r_coord):
    return jax.jacfwd(psi, argnums=0)(psi_params, r_coord)


def get_delta_params(
        psi
        , psi_params
        , psi_vector
        , r_coords
        , particle_pairs
        , particle_triplets
        , spin_exchange_indices
        , learning_rate
        , hamiltonian=hamiltonian_psi
        , include_sr_equations=True
        , return_loss=False):
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
    param_axis = 1
    spin_axis = -1
    walker_axis = 0
    d_psi = vmap(d_psi_d_params, in_axes=(None, None, walker_axis))(psi, psi_params,
                                                                    r_coords)  # [n_walkers, psi_output, n_params]
    d_psi = jnp.moveaxis(d_psi, -1, param_axis)
    d_psi = jnp.tensordot(d_psi, psi_vector, axes=walker_axis)  # walkers, params, spin-isospin
    h_psi = vmap(hamiltonian, in_axes=(None, None, None, walker_axis, None, None, None))(
        psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets,
        spin_exchange_indices)  # [n_walkers, psi_out]
    psi_r = vmap(psi, in_axes=(None, walker_axis))(psi_params, r_coords)
    psi_r = jnp.tensordot(psi_r, psi_vector, axes=walker_axis)

    psi_psi = vmap(jnp.vdot, in_axes=(walker_axis, walker_axis))(psi_r, psi_r)
    d_psi_h_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(walker_axis, walker_axis))(x, y),
                       in_axes=(param_axis, None))(d_psi, h_psi)
    d_psi_h_psi = jnp.moveaxis(d_psi_h_psi, walker_axis, param_axis)
    d_psi_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(walker_axis, walker_axis))(x, y), in_axes=(param_axis, None))(
        d_psi, psi_r)
    d_psi_psi = jnp.moveaxis(d_psi_psi, walker_axis, param_axis)
    psi_h_psi = vmap(jnp.vdot, in_axes=(walker_axis, walker_axis))(psi_r, h_psi)
    d_psi_psi_avg = (d_psi_psi / psi_psi[:, None]).mean(axis=walker_axis)
    loss = (psi_h_psi / psi_psi).mean(axis=walker_axis)
    d_energy = 2.0 * (d_psi_h_psi / psi_psi[:, None]).mean(axis=walker_axis) - 2.0 * loss * d_psi_psi_avg
    delta_p = - learning_rate * d_energy

    if include_sr_equations:
        d_psi_d_psi = \
            vmap(
                vmap(
                    vmap(
                        jnp.vdot  # dot prod over spin
                        , in_axes=(None, param_axis - 1))
                    , in_axes=(param_axis - 1, None))
                , in_axes=(walker_axis, walker_axis))(d_psi, d_psi)  # walkers, params, params
        d_psi_d_psi_avg = (d_psi_d_psi / psi_psi[:, None, None]).mean(axis=walker_axis)
        psi_d_psi_avg = (jnp.conj(d_psi_psi) / psi_psi[:, None]).mean(axis=walker_axis)
        d_psi_psi_avg_psi_d_psi_avg = jnp.tensordot(d_psi_psi_avg, psi_d_psi_avg, axes=walker_axis)  # params, params
        S_ij = d_psi_d_psi_avg - d_psi_psi_avg_psi_d_psi_avg

        eps = 0.0001
        S_ij += eps * jnp.identity(S_ij.shape[0])
        cho_factor_solution = cho_factor(S_ij)
        delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)

    if return_loss:
        return delta_p, loss
    else:
        return delta_p
