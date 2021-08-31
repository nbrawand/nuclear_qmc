from jax import numpy as jnp


def get_local_energy(psi, psi_params, r_coords, hamiltonian):
    """

    Parameters
    ----------
    psi: function
        The prefactor of the wave function taking two arguments psi_params and array of particle coordinates.
    psi_params: ndarray
        1D array containing wave function parameters.
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.
    hamiltonian: function
        Returns H|psi> given `psi`, `psi_params`, `r_coords`.

    Returns
    -------
    float
        The local energy evaluated at `r_coords`.

    """
    h_psi = hamiltonian(psi, psi_params, r_coords)
    psi_r = psi(psi_params, r_coords)
    psi_psi = jnp.real(jnp.vdot(psi_r, psi_r))
    psi_h_psi = jnp.real(jnp.vdot(psi_r, h_psi))
    return psi_h_psi / psi_psi
