import jax
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from nuclear_qmc.operators.hamiltonian import hamiltonian_psi


def d_psi_d_params(psi, psi_params, r_coord):
    return jax.grad(psi, argnums=0)(psi_params, r_coord)


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

    d_psi = vmap(d_psi_d_params, in_axes=(None, None, 0))(psi, psi_params, r_coords)
    d_psi = jnp.tensordot(d_psi, psi_vector, axes=0)  # walkers, params, spin-isospin
    h_psi = vmap(hamiltonian, in_axes=(None, None, None, 0, None, None, None))(
        psi, psi_params, psi_vector, r_coords, particle_pairs, particle_triplets, spin_exchange_indices)
    psi_r = vmap(psi, in_axes=(None, 0))(psi_params, r_coords)
    psi_r = jnp.tensordot(psi_r, psi_vector, axes=0)

    psi_psi = vmap(jnp.vdot, in_axes=(0, 0))(psi_r, psi_r)
    d_psi_h_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(0, 0))(x, y), in_axes=(1, None))(d_psi, h_psi)
    d_psi_psi = vmap(lambda x, y: vmap(jnp.vdot, in_axes=(0, 0))(x, y), in_axes=(1, None))(d_psi, psi_r)
    psi_h_psi = vmap(jnp.vdot, in_axes=(0, 0))(psi_r, h_psi)
    d_psi_psi_avg = (d_psi_psi / psi_psi).mean(axis=1)
    loss = (psi_h_psi / psi_psi).mean(axis=0)
    d_energy = 2.0 * (d_psi_h_psi / psi_psi).mean(axis=1) - 2.0 * loss * d_psi_psi_avg
    delta_p = - learning_rate * d_energy

    if include_sr_equations:
        d_psi_d_psi = \
            vmap(
                vmap(
                    vmap(
                        jnp.vdot
                        , in_axes=(None, 0))
                    , in_axes=(0, None))
                , in_axes=(0, 0))(d_psi, d_psi)  # walkers, params, spin-isospin
        d_psi_d_psi_avg = (d_psi_d_psi / jnp.expand_dims(psi_psi, axis=(1, 2))).mean(axis=0)
        psi_d_psi_avg = (jnp.conj(d_psi_psi) / psi_psi).mean(axis=1)
        d_psi_psi_avg_psi_d_psi_avg = jnp.tensordot(d_psi_psi_avg, psi_d_psi_avg, axes=0)
        S_ij = d_psi_d_psi_avg - d_psi_psi_avg_psi_d_psi_avg

        eps = 0.0001
        S_ij += eps * jnp.identity(S_ij.shape[0])
        cho_factor_solution = cho_factor(S_ij)
        delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)

    if return_loss:
        return delta_p, loss
    else:
        return delta_p
