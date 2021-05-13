import jax
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from nuclear_qmc.operators.hamiltonian import hamiltonian_psi


def get_d_psi_psi(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi
    psi_params
    r_coord: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    d_psi = get_d_psi(psi, psi_params, psi_vector, r_coords)  # [n_params, spin_isospin]
    psi_r = get_psi_r(psi, psi_params, psi_vector, r_coords)
    return vmap(jnp.vdot, in_axes=(0, None))(d_psi, psi_r)  # [n_params]


def get_psi_psi(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    psi_r = get_psi_r(psi, psi_params, psi_vector, r_coords)
    return jnp.vdot(psi_r, psi_r)


def get_psi_r(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    psi_r = psi(psi_params, r_coords) * psi_vector
    return psi_r


def get_psi_h_psi(psi
                  , psi_params
                  , psi_vector
                  , r_coords
                  , particle_pairs
                  , particle_triplets
                  , spin_exchange_indices
                  , hamiltonian):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]
    particle_pairs
    particle_triplets
    spin_exchange_indices

    Returns
    -------
    \\sum_s \\Psi_s(R) H \\Psi_s(R)

    """
    h_psi = hamiltonian(psi
                        , psi_params
                        , psi_vector
                        , r_coords
                        , particle_pairs
                        , particle_triplets
                        , spin_exchange_indices)
    psi_r = psi(psi_params, r_coords) * psi_vector
    psi_h_psi = jnp.vdot(psi_r, h_psi)
    return psi_h_psi


def get_d_psi_h_psi(psi
                    , psi_params
                    , psi_vector
                    , r_coords
                    , particle_pairs
                    , particle_triplets
                    , spin_exchange_indices
                    , hamiltonian):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]
    particle_pairs
    particle_triplets
    spin_exchange_indices

    Returns
    -------
    \\sum_s \\Psi_s(R) H \\Psi_s(R)

    """
    h_psi = hamiltonian(psi
                        , psi_params
                        , psi_vector
                        , r_coords
                        , particle_pairs
                        , particle_triplets
                        , spin_exchange_indices)  # [spin_isospin]
    d_psi = get_d_psi(psi, psi_params, psi_vector, r_coords)  # [n_params, spin_isospin]
    return vmap(jnp.vdot, in_axes=(0, None))(d_psi, h_psi)  # [n_params]


def get_d_psi(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    d_psi = jax.jacfwd(psi, argnums=0)(psi_params, r_coords)  # [psi_output_dimensions, n_params]
    d_psi = jnp.moveaxis(d_psi, 0,
                         -1)  # [n_params, si_output_dimensions]  This is necessary if psi has the spin or if psi_vector has the spin
    d_psi = jnp.tensordot(d_psi, psi_vector, axes=0)  # [n_params, n_spin_isospin]
    return d_psi


def get_d_psi_d_psi(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi
    psi_params
    psi_vector
    r_coords: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    d_psi = get_d_psi(psi, psi_params, psi_vector, r_coords)  # [n_params, spin_isospin]
    d_psi_d_psi = \
        vmap(
            vmap(
                jnp.vdot  # dot prod over spin of each params_i * param_j
                , in_axes=(None, 0))
            , in_axes=(0, None))(d_psi, d_psi)  # double for loop over params
    return d_psi_d_psi  # [n_params, n_params]


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
    walker_axis = 0
    psi_psi = vmap(get_psi_psi, in_axes=(None, None, None, walker_axis))(psi
                                                                         , psi_params
                                                                         , psi_vector
                                                                         , r_coords)  # [n_walkers]
    psi_h_psi = vmap(get_psi_h_psi, in_axes=(None, None, None, walker_axis, None, None, None, None))(psi
                                                                                                     , psi_params
                                                                                                     , psi_vector
                                                                                                     , r_coords
                                                                                                     , particle_pairs
                                                                                                     , particle_triplets
                                                                                                     ,
                                                                                                     spin_exchange_indices
                                                                                                     ,
                                                                                                     hamiltonian)  # [n_walkers]
    psi_h_psi_avg = (psi_h_psi / psi_psi).mean(axis=walker_axis)
    del psi_h_psi
    d_psi_psi = vmap(get_d_psi_psi, in_axes=(None, None, None, walker_axis))(psi
                                                                             , psi_params
                                                                             , psi_vector
                                                                             , r_coords)  # [n_walkers, params]
    d_psi_psi_avg = (d_psi_psi / psi_psi[:, None]).mean(axis=walker_axis)
    del d_psi_psi
    d_psi_h_psi = vmap(get_d_psi_h_psi, in_axes=(None, None, None, walker_axis, None, None, None, None))(psi
                                                                                                         , psi_params
                                                                                                         , psi_vector
                                                                                                         , r_coords
                                                                                                         ,
                                                                                                         particle_pairs
                                                                                                         ,
                                                                                                         particle_triplets
                                                                                                         ,
                                                                                                         spin_exchange_indices
                                                                                                         , hamiltonian)
    d_psi_h_psi_avg = (d_psi_h_psi / psi_psi[:, None]).mean(axis=walker_axis)
    del d_psi_h_psi
    d_energy = 2.0 * d_psi_h_psi_avg - 2.0 * psi_h_psi_avg * d_psi_psi_avg
    delta_p = - learning_rate * d_energy

    if include_sr_equations:
        d_psi_d_psi = vmap(get_d_psi_d_psi, in_axes=(None, None, None, walker_axis))(psi
                                                                                     , psi_params
                                                                                     , psi_vector
                                                                                     ,
                                                                                     r_coords)  # walkers, params, params
        d_psi_d_psi_avg = (d_psi_d_psi / psi_psi[:, None, None]).mean(axis=walker_axis)
        del d_psi_d_psi
        del psi_psi
        psi_d_psi_avg = jnp.conj(d_psi_psi_avg)
        d_psi_psi_avg_psi_d_psi_avg = jnp.tensordot(d_psi_psi_avg, psi_d_psi_avg, axes=0)  # params, params
        S_ij = d_psi_d_psi_avg - d_psi_psi_avg_psi_d_psi_avg

        eps = 0.0001
        S_ij += eps * jnp.identity(S_ij.shape[0])
        cho_factor_solution = cho_factor(S_ij)
        delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)

    if return_loss:
        return delta_p, psi_h_psi_avg
    else:
        return delta_p
