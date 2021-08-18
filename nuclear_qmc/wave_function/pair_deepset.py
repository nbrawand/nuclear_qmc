import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Softplus, Tanh, Sigmoid, elementwise)
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.ops import index, index_add, index_update

from functools import partial
import pickle


@jit
def lintanh(x):
    return x - jnp.tanh(x) / 2


Lintanh = elementwise(lintanh)
Sin = elementwise(jnp.sin)


class PairDeepset(object):
    def __init__(self, ndim, npart, conf, key, mix, spin):
        self.ndim = ndim
        self.npart = npart
        self.conf = conf
        self.key = key
        self.mix = mix
        self.spin = spin
        self.ip = spin[:, 0]
        self.jp = spin[:, 1]
        self.npair = len(spin)
        self.k = jnp.arange(self.npair)
        self.ndense = 6
        self.nlat = 4  ## 1 * (self.ndim * self.npart + 4)
        #        self.activation = Tanh
        self.activation = Tanh
        self.a = 8

    def build(self):
        # phi_a
        self.phi_a_init, self.phi_a_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(self.nlat),
        )
        in_shape = (-1, 2 * self.ndim)
        self.key, key_input = jax.random.split(self.key)
        phi_a_shape, phi_a_params = self.phi_a_init(key_input, in_shape)
        self.num_phi_a_params = len(phi_a_params)

        # rho_a
        self.rho_a_init, self.rho_a_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(1),
        )
        self.key, key_input = jax.random.split(self.key)
        rho_a_shape, rho_a_params = self.rho_a_init(key_input, phi_a_shape)
        self.num_rho_a_params = len(rho_a_params)

        self.key, key_input = jax.random.split(self.key)
        theta_init = normal()
        theta = [theta_init(key_input, (1,))]

        net_params = phi_a_params + rho_a_params + theta

        # cast to double
        net_params = jax.tree_multimap(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)

        return flat_net_params

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r):
        params = self.unflatten_params(params)

        rcm = jnp.mean(r, axis=0)
        r = r - rcm[None, :]
        x = r

        def x_pair(k, x):
            x_ij = jnp.concatenate((x[self.ip[k], :], x[self.jp[k], :]))
            x_ji = jnp.concatenate((x[self.jp[k], :], x[self.ip[k], :]))
            return x_ij, x_ji

        x_ij, x_ji = vmap(x_pair, in_axes=(0, None))(self.k, x)
        x_ij = jnp.append(x_ij, x_ji, axis=0)
        num_offset_params = 0
        phi_a_params = params[num_offset_params: num_offset_params + self.num_phi_a_params]
        num_offset_params = num_offset_params + self.num_phi_a_params
        rho_a_params = params[num_offset_params: num_offset_params + self.num_rho_a_params]
        num_offset_params = num_offset_params + self.num_rho_a_params
        theta_params = params[num_offset_params:]
        phi_a_out = jnp.mean(vmap(self.phi_a_apply, in_axes=(None, 0))(phi_a_params, x_ij), axis=0)
        rho_a_out = self.rho_a_apply(rho_a_params, phi_a_out)
        theta = theta_params[0]
        rho_a_out = self.a * jnp.tanh((rho_a_out - theta) / self.a) + theta
        psi = jnp.exp(rho_a_out)
        return jnp.reshape(psi, ())

    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters

    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters

    @staticmethod
    def update_cast(params):
        return params.astype(jnp.float64)
