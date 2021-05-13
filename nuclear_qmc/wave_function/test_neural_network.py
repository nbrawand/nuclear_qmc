import jax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
import pickle
import os

from nuclear_qmc.wave_function.utility import apply_confining_potential

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_params(params_file_name):
    with open(params_file_name, 'rb') as fil:
        return pickle.load(fil)


def save(params, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(params, file)


def build_test_nn_wfc(key=None, params_file=os.path.join(dir_path, 'test_neural_network.model')):
    if key is None:
        key = random.PRNGKey(0)
    ndense = 8
    activation = Tanh
    phi_a_init, phi_a_apply = stax.serial(
        Dense(ndense), activation,
        Dense(ndense), Tanh,
        Dense(1),
    )
    in_shape = (1,)

    if params_file is not None:
        unflattened_params = load_params(params_file)
    else:
        key, key_input = jax.random.split(key)
        _, unflattened_params = phi_a_init(key_input, in_shape)

    flat_params, unflatten_params_function = ravel_pytree(unflattened_params)

    def psi_prefactor(flat_params_in, r_coords):
        rcm = jnp.mean(r_coords, axis=0)
        r = r_coords - rcm[None, :]
        delta_r = jnp.linalg.norm(r[0, :] - r[1, :])
        unflat_params = unflatten_params_function(flat_params_in)
        phi_a_out = phi_a_apply(unflat_params, delta_r)
        phi_a_out = jnp.mean(phi_a_out)
        psi = jnp.exp(phi_a_out)
        psi *= apply_confining_potential(r)
        return jnp.reshape(psi, ())

    return key, psi_prefactor, flat_params
