from jax import numpy as jnp
from nuclear_qmc.wave_function.neural_network_jastro_builder.build_nn_wave_function import build_nn_wave_function


def build_radial_function(key
                          , n_dense
                          , n_hidden_layers
                          , nn_wrapper_function=jnp.exp
                          ):
    key, nn, params = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)

    def func(p, r_i):
        r_mag = jnp.linalg.norm(r_i)
        out = nn(p, r_mag)
        out = nn_wrapper_function(out)
        return out

    return key, func, params
