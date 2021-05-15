import jax
from jax import random
from jax.flatten_util import ravel_pytree
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
import pickle


def load_params(params_file_name):
    with open(params_file_name, 'rb') as fil:
        return pickle.load(fil)


def build_nn_wfc(ndense=4, key=None, params_file=None, in_shape=(1,)):
    if key is None:
        key = random.PRNGKey(0)

    activation = Tanh
    phi_a_init, phi_a_apply = stax.serial(
        Dense(ndense), activation,
        Dense(ndense), Tanh,
        Dense(1),
    )

    if params_file is not None:
        unflattened_params = load_params(params_file)
    else:
        key, key_input = jax.random.split(key)
        _, unflattened_params = phi_a_init(key_input, in_shape)

    flat_params, unflatten_params_function = ravel_pytree(unflattened_params)

    def psi_prefactor(flat_params_in, nn_input):
        unflat_params = unflatten_params_function(flat_params_in)
        psi_out = phi_a_apply(unflat_params, nn_input)
        return psi_out

    return key, psi_prefactor, flat_params
