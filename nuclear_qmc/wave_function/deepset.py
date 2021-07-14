from nuclear_qmc.wave_function.jastro_neural_network_builder.build_nn_wave_function import build_nn_wave_function

import jax.numpy as jnp
import jax


def get_deep_set(key, n_dense, n_hidden_layers, in_shape, out_shape):
    key, nn1, params1 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=in_shape, output_size=out_shape)
    n_p = len(params1)
    key, nn2, params2 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=(out_shape,), output_size=1)

    def psi(params, r):
        latent = jax.vmap(nn1, in_axes=(None, 0))(params[:n_p], r)
        latent = latent.sum(axis=0)
        out = nn2(params[n_p:], latent)
        return out

    params = jnp.concatenate((params1, params2))

    return key, psi, params
