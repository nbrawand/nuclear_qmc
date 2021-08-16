from nuclear_qmc.wave_function.neural_network_jastro_builder.build_nn_wave_function import build_nn_wave_function

import jax.numpy as jnp
import jax


def get_deepset_jastro(key, n_dense, n_hidden_layers, out_shape, in_shape=(3,), latent_shape=6, wrapper_func=jnp.tanh):
    key, nn1, params1 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=in_shape, output_size=latent_shape, reshape_output=False)
    n_p = len(params1)
    key, nn2, params2 = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers,
                                               in_shape=(latent_shape,), output_size=out_shape, reshape_output=False)

    def psi(params, r):
        latent = jax.vmap(nn1, in_axes=(None, 0))(params[:n_p], r)
        latent = latent.mean(axis=0)
        out = nn2(params[n_p:], latent)
        out = wrapper_func(out)
        return out

    params = jnp.concatenate((params1, params2))

    return key, psi, params
