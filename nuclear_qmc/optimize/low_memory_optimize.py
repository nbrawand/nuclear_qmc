import jax
from jax.lax import fori_loop
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def get_d_psi_psi(d_psi, psi_r):
    return vmap(jnp.vdot, in_axes=(0, None))(d_psi, psi_r)  # [n_params]


def get_psi_r(psi, psi_params, psi_vector, r_coords):
    """

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords: ndarray
        [n_particles, n_dimensions] coordinates of the particles.

    Returns
    -------
    ndarray
        [spin_isospin] psi evaluated at `r_coords`.

    """
    psi_r = psi(psi_params, r_coords) * psi_vector
    return psi_r


def get_d_psi_h_psi(d_psi, h_psi):
    return vmap(jnp.vdot, in_axes=(0, None))(d_psi, h_psi)  # [n_params]


def get_d_psi(psi, psi_params, psi_vector, r_coords):
    """

    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords: ndarray
        [n_particles, n_dimensions] coordinates of the particles.

    Returns
    -------
    ndarray
        [n_params, n_spin_isospin] :math:`\\partial_i \\psi`

    """
    d_psi = jax.jacfwd(psi, argnums=0)(psi_params, r_coords)  # [psi_output_dimensions, n_params]
    d_psi = jnp.moveaxis(d_psi, -1, 0)  # make [n_params, si_output_dimensions]
    d_psi = jnp.tensordot(d_psi, psi_vector, axes=0)  # [n_params, n_spin_isospin]
    return d_psi


def get_d_psi_d_psi(d_psi):
    """

    Parameters
    ----------

    Returns
    -------
    ndarray
        [n_params, n_params] :math:`\\partial_i \\psi \\partial_j \\psi`

    """
    d_psi_d_psi = \
        vmap(
            vmap(
                jnp.vdot  # dot prod over spin of each params_i * param_j
                , in_axes=(None, 0))
            , in_axes=(0, None))(d_psi, d_psi)  # double for loop over params
    return d_psi_d_psi  # [n_params, n_params]


def get_d_psi_d_psi_avg(d_psi, psi_psi, psi_param_len):
    """

    Parameters
    ----------

    Returns
    -------
    ndarray
        [n_params, n_params] :math:`\\langle \\partial_i \\psi | \\partial_j \\psi \\rangle`.

    """

    def sum_d_psi_d_psi(i, d_psi_d_psi_avg):
        d_psi_d_psi = get_d_psi_d_psi(d_psi[i])  # [n_param, n_param]
        d_psi_d_psi_avg += d_psi_d_psi / psi_psi[i]
        return d_psi_d_psi_avg

    d_psi_d_psi_avg = jnp.zeros(shape=(psi_param_len, psi_param_len))
    d_psi_d_psi_avg = fori_loop(0, len(psi_psi), sum_d_psi_d_psi, d_psi_d_psi_avg)
    d_psi_d_psi_avg /= len(psi_psi)
    return d_psi_d_psi_avg


def get_delta_params(
        psi
        , psi_params
        , psi_vector
        , r_coords
        , learning_rate
        , hamiltonian
        , include_sr_equations=True
        , return_loss=False, eps=0.0001):
    """Calculate `psi_params` update by solving stochastic reconfiguration equations.

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    psi_vector: ndarray
        2D array containing wave function spin isospin components.
    r_coords: ndarray
        [n_walkers, n_particles, n_dimensions] coordinates of the walkers.
    particle_pairs: ndarray
        [n_pairs, 2] particle indices for each pair.
    particle_triplets: ndarray
        [n_triplets, 3] particle indices for each pair.
    spin_exchange_indices:
        2D array containing the indices after applying :math:`\\sigma_{ij}` to `psi_vector`.
    learning_rate: float, optional
        Size of learning rate for updating `psi_params`.
    hamiltonian: function, optional
        The hamiltonian function to apply to the wave function.
    include_sr_equations: bool
        If True solve stochastic reconfiguration equations else just to stochastic gradient descent.
    return_loss: bool
        Return total energy at end of calculation with parameter update.
    eps: float, optional
        Size of diagonal for stabilizing the stochastic reconfiguration equations. Reasonable values are between 10^4
        and 10^6.

    Returns
    -------
    delta_p: ndarray
        1D array same size as `psi_params`. Add `delta_p` to `psi_params` to get updated parameters.

    """
    walker_axis = 0
    h_psi = vmap(hamiltonian, in_axes=(None, None, None, walker_axis))(psi
                                                                       , psi_params
                                                                       , psi_vector
                                                                       , r_coords)
    # n_walkers n_spin_isospin
    psi_r = vmap(get_psi_r, in_axes=(None, None, None, walker_axis))(psi, psi_params, psi_vector, r_coords)
    psi_psi = vmap(lambda x: jnp.vdot(x, x))(psi_r)  # [n_walkers]
    psi_h_psi = vmap(lambda p, h: jnp.vdot(p, h))(psi_r, h_psi)  # n_walkers
    psi_h_psi_avg = (psi_h_psi / psi_psi).mean(axis=walker_axis)
    # [n_walkers, n_params, spin_isospin]
    d_psi = vmap(get_d_psi, in_axes=(None, None, None, walker_axis))(psi, psi_params, psi_vector, r_coords)
    d_psi_psi = vmap(get_d_psi_psi)(d_psi, psi_r)
    d_psi_psi_avg = (d_psi_psi / psi_psi[:, None]).mean(axis=walker_axis)
    d_psi_h_psi = vmap(get_d_psi_h_psi)(d_psi, h_psi)
    d_psi_h_psi_avg = (d_psi_h_psi / psi_psi[:, None]).mean(axis=walker_axis)
    d_energy = 2.0 * d_psi_h_psi_avg - 2.0 * psi_h_psi_avg * d_psi_psi_avg
    delta_p = - learning_rate * d_energy

    if include_sr_equations:
        d_psi_d_psi_avg = get_d_psi_d_psi_avg(d_psi, psi_psi, len(psi_params))  # params, params
        psi_d_psi_avg = jnp.conj(d_psi_psi_avg)
        d_psi_psi_avg_psi_d_psi_avg = jnp.tensordot(d_psi_psi_avg, psi_d_psi_avg, axes=0)  # params, params
        S_ij = d_psi_d_psi_avg - d_psi_psi_avg_psi_d_psi_avg
        S_ij += eps * jnp.identity(S_ij.shape[0])
        cho_factor_solution = cho_factor(S_ij)
        delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)

    if return_loss:
        return delta_p, psi_h_psi_avg
    else:
        return delta_p
