import jax
from jax import vmap, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from nuclear_qmc.operators.hamiltonian import hamiltonian_psi, get_local_energy
from nuclear_qmc.operators.operators import kinetic_energy_psi
from nuclear_qmc.wave_function.wave_function import WaveFunction


def partial_full_psi_parameters(wave_function: WaveFunction, r_coords):
    """ \partial_{params} \vec{\psi_{params}(r)}

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]

    Returns
    -------

    """
    psi = lambda r, p, s: wave_function.psi_prefactor(r, p) * wave_function.psi_vector(r, p, s)
    return jax.jacrev(psi, argnums=1)(r_coords, wave_function.params, wave_function.spin)


def partial_psi_prefactor_parameters(wave_function: WaveFunction, r_coords):
    """ \partial_{params} \psi_{params}(r)

    Parameters
    ----------
    wave_function: WaveFunction
    r_coords: ndarray[n_particles, n_dimensions]

    Returns
    -------

    """
    return jax.grad(wave_function.psi_prefactor, argnums=1)(r_coords, wave_function.params)


def get_new_wave_function_parameters(wave_function: WaveFunction
                                     , r_coords
                                     , learning_rate
                                     , partial_function=partial_full_psi_parameters
                                     , kinetic_energy_operator=kinetic_energy_psi):
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
    #  calculate necessary components and average over walkers
    d_psi = vmap(partial_function, in_axes=(None, 0))(wave_function, r_coords).mean(
        axis=0)  # [n_spin, n_params]
    h_psi = vmap(hamiltonian_psi, in_axes=(None, 0, None))(wave_function, r_coords, kinetic_energy_operator).mean(
        axis=0)  # [n_spin]
    psi_r = vmap(wave_function.psi, in_axes=(0,))(r_coords).mean(axis=0)  # [n_spin]

    # energy derivative
    psi_psi = jnp.vdot(psi_r, psi_r)  # float
    d_psi_h_psi = vmap(jnp.vdot, in_axes=(2, None))(d_psi, h_psi)  # [n_params]
    d_psi_psi = vmap(jnp.vdot, in_axes=(2, None))(d_psi, psi_r)  # [n_params]
    psi_h_psi = jnp.vdot(psi_r, h_psi)  # float
    d_energy = 2.0 * (d_psi_h_psi - psi_h_psi * d_psi_psi) / psi_psi  # [n_params]

    # return wave_function.params -learning_rate * d_energy

    # quantum fisher information
    def vdot_nested(in_d_psi):
        return vmap(jnp.vdot, in_axes=(2, None))(d_psi, in_d_psi)

    d_psi_d_psi = vmap(vdot_nested, in_axes=(2,))(d_psi)  # [n_params, n_params]
    fisher_information = d_psi_d_psi / psi_psi
    fisher_information -= jnp.tensordot(d_psi_psi, d_psi_psi, axes=0) / psi_psi ** 2

    max_eps = 0.1
    min_eps = 0.0000001
    energy_change_min = 1.0
    out_params = wave_function.params
    for n in range(4):
        eps = (max_eps + min_eps) / 2.0
        small_diag_matrix = eps * jnp.identity(fisher_information.shape[0])
        fisher_information = small_diag_matrix + fisher_information
        cho_factor_solution = cho_factor(fisher_information)
        delta_p = cho_solve(cho_factor_solution, -learning_rate * d_energy)

        original_local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coords)
        original_local_energy_mean = original_local_energy.mean()
        original_local_energy_std = original_local_energy.std()

        wave_function.params += delta_p
        new_local_energy = vmap(get_local_energy, in_axes=(None, 0))(wave_function, r_coords)
        new_local_energy_mean = new_local_energy.mean()
        new_local_energy_std = new_local_energy.std()
        wave_function.params -= delta_p

        energy_change = new_local_energy_mean - original_local_energy_mean
        energy_std_change = new_local_energy_std - original_local_energy_std
        if energy_change < energy_change_min and energy_std_change < 1.:
            energy_change_min = energy_change
            out_params = wave_function.params + delta_p
            max_eps = eps
        else:
            min_eps = eps

    return out_params
