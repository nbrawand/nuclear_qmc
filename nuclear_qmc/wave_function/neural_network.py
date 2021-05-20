import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
import pickle


def load_params(params_file_name):
    with open(params_file_name, 'rb') as fil:
        return pickle.load(fil)


def build_nn_wfc(ndense=4, key=None, params_file=None, in_shape=(1,), n_hidden_layers=2,
                 dtype=jnp.float64):
    if key is None:
        key = random.PRNGKey(0)

    hidden_layers = n_hidden_layers * [Dense(ndense), Tanh, ]
    phi_a_init, phi_a_apply = stax.serial(
        *hidden_layers,
        Dense(1),
    )

    if params_file is not None:
        unflattened_params = load_params(params_file)
    else:
        key, key_input = jax.random.split(key)
        _, unflattened_params = phi_a_init(key_input, in_shape)
        unflattened_params = jax.tree_multimap(lambda params: params.astype(jnp.float64), unflattened_params)

    flat_params, unflatten_params_function = ravel_pytree(unflattened_params)

    def psi_prefactor(flat_params_in, nn_input):
        unflat_params = unflatten_params_function(flat_params_in)
        psi_out = phi_a_apply(unflat_params, nn_input)
        return psi_out.reshape(-1)

    return key, psi_prefactor, flat_params
