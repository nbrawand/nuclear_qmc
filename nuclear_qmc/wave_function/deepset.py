from nuclear_qmc.wave_function.jastro_neural_network_builder.build_nn_wave_function import build_nn_wave_function
from nuclear_qmc.utils.get_dr_ij import get_r_ij

import jax.numpy as jnp
import jax


def get_deep_set(key, n_dense, n_hidden_layers, out_shape, in_shape=(3,), latent_shape=6, wrapper_func=jnp.tanh):
    key, nn1, params1 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=in_shape, output_size=latent_shape, reshape_output=False)
    n_p = len(params1)
    key, nn2, params2 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=(latent_shape,), output_size=out_shape, reshape_output=True)

    def psi(params, r):
        latent = jax.vmap(nn1, in_axes=(None, 0))(params[:n_p], r)
        latent = latent.sum(axis=0)
        out = nn2(params[n_p:], latent)
        out = wrapper_func(out)
        return out

    params = jnp.concatenate((params1, params2))

    return key, psi, params


def get_deepset_jastro(key, n_dense, n_hidden_layers, particle_pairs, latent_shape):
    key, deepset_net, deepset_params = get_deep_set(key, n_dense, n_hidden_layers
                                                    , out_shape=1
                                                    , in_shape=(2,)
                                                    , latent_shape=latent_shape
                                                    , wrapper_func=jnp.exp)

    def deepset_func(params, r_coords):
        r_ij = get_r_ij(r_coords, particle_pairs, jnp.subtract)
        r_p_ij = get_r_ij(r_coords, particle_pairs, jnp.add)
        x = jnp.column_stack((r_ij, r_p_ij))
        out = deepset_net(params, x)
        return out

    return key, deepset_func, deepset_params
