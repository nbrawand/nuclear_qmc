import jax.numpy as jnp
from jax import vmap
from nuclear_qmc.utils.center_particles import center_particles
from nuclear_qmc.wave_function.neural_network import build_nn_wfc
from nuclear_qmc.wave_function.utility import apply_confining_potential


def build_deep_set(key, n_dense, n_latent, n_dimensions):
    key, phi_net, phi_params = build_nn_wfc(ndense=n_dense
                                            , key=key
                                            , in_shape=(n_dimensions,)
                                            , out_shape=n_latent)
    key, rho_net, rho_params = build_nn_wfc(ndense=n_dense
                                            , key=key
                                            , in_shape=(n_latent,)
                                            , out_shape=1)
    n_phi_params = len(phi_params)
    params = jnp.concatenate((phi_params, rho_params))

    def psi_function(in_params, r_coords):
        r_coords = center_particles(r_coords)
        phi = vmap(phi_net, in_axes=(None, 0))(in_params[:n_phi_params], r_coords)
        phi = jnp.mean(phi, axis=0)
        rho = rho_net(in_params[n_phi_params:], phi)
        psi = jnp.exp(rho)
        psi *= apply_confining_potential(r_coords)
        return psi

    return key, psi_function, params
