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


def read(fil):
    with open(fil, 'rb') as file:
        params_read = pickle.load(file)
    return params_read


@jit
def lintanh(x):
    return x - jnp.tanh(x) / 2


Lintanh = elementwise(lintanh)
Sin = elementwise(jnp.sin)


class Wavefunction(object):
    def __init__(self, ndim, npart, conf, key, mix, spin, ip, jp):
        self.ndim = ndim
        self.npart = npart
        self.conf = conf
        self.key = key
        self.mix = mix
        self.spin = spin
        self.ip = ip
        self.jp = jp
        self.npair = len(ip)
        self.k = jnp.arange(npart)
        self.ndense = 12
        self.nlat = 12  ## 1 * (self.ndim * self.npart + 4)
        #        self.activation = Tanh
        self.activation = Tanh
        self.a = 8

    def build(self):
        # phi_a
        self.phi_a_init, self.phi_a_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(self.nlat),
        )
        in_shape = (-1, 2 * (self.ndim ))
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

        ## phi_p
        #        self.phi_p_init, self.phi_p_apply = stax.serial(
        #        Dense(self.ndense), self.activation,
        #        Dense(self.nlat),
        # )
        #        in_shape = (-1, ( self.ndim + 2 ) )
        #        self.key, key_input = jax.random.split(self.key)
        #        phi_p_shape, phi_p_params = self.phi_p_init(key_input,in_shape)
        #        self.num_phi_p_params = len(phi_p_params)
        #
        ## rho_p
        #        self.rho_p_init, self.rho_p_apply = stax.serial(
        #        Dense(self.ndense), self.activation,
        #        Dense(1),
        # )
        #        self.key, key_input = jax.random.split(self.key)
        #        rho_p_shape, rho_p_params = self.rho_p_init(key_input,phi_p_shape)
        #        self.num_rho_p_params = len(rho_p_params)

        # Rn0
        self.R_n0_init, self.R_n0_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(self.ndense), Tanh,
            Dense(1),
        )
        in_shape = (-1, 1)
        self.key, key_input = jax.random.split(self.key)
        R_n0_shape, R_n0_params = self.R_n0_init(key_input, in_shape)
        self.num_R_n0_params = len(R_n0_params)

        # Rn1
        self.R_n1_init, self.R_n1_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(self.ndense), Tanh,
            Dense(1),
        )
        in_shape = (-1, 1)
        self.key, key_input = jax.random.split(self.key)
        R_n1_shape, R_n1_params = self.R_n1_init(key_input, in_shape)
        self.num_R_n1_params = len(R_n1_params)

        # theta
        self.key, key_input = jax.random.split(self.key)
        theta_init = normal()
        theta = [theta_init(key_input, (1,))]

        #        net_params = phi_a_params + rho_a_params +  phi_p_params + rho_p_params + R_n0_params + R_n1_params + theta

        net_params = phi_a_params + rho_a_params + R_n0_params + R_n1_params + theta

        # cast to double
        net_params = jax.tree_multimap(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)
        num_flat_params = flat_net_params.shape[0]

        with open('full.model', 'wb') as file:
            pickle.dump(net_params, file)

        return net_params

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r):
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

        #        phi_p_params = params[num_offset_params : num_offset_params + self.num_phi_p_params]
        #        num_offset_params = num_offset_params + self.num_phi_p_params
        #        rho_p_params = params[num_offset_params : num_offset_params + self.num_rho_p_params]
        #        num_offset_params = num_offset_params + self.num_rho_p_params

        R_n0_params = params[num_offset_params: num_offset_params + self.num_R_n0_params]
        num_offset_params = num_offset_params + self.num_R_n0_params
        R_n1_params = params[num_offset_params: num_offset_params + self.num_R_n1_params]
        num_offset_params = num_offset_params + self.num_R_n1_params
        theta_params = params[num_offset_params:]

        phi_a_out = jnp.mean(vmap(self.phi_a_apply, in_axes=(None, 0))(phi_a_params, x_ij), axis=0)
        rho_a_out = self.rho_a_apply(rho_a_params, phi_a_out)
        theta = theta_params[0]
        rho_a_out = self.a * jnp.tanh((rho_a_out - theta) / self.a) + theta

        #        phi_p_out = jnp.mean(vmap(self.phi_p_apply, in_axes=(None,0))( phi_p_params, x ), axis=0)
        #        rho_p_out = self.rho_a_apply(rho_p_params, phi_p_out)

        psi = jnp.exp(rho_a_out)  # * jnp.tanh(rho_p_out)

        # R_nl = vmap(self.R_nl_sp, in_axes=(0, None, None))(r, R_n0_params, R_n1_params)

        # Y_1m = vmap(self.Y_1m_sp, in_axes=(0))(r)

        # psi *= self.spin.phi_spin(sz, R_nl, Y_1m)

        return jnp.reshape(psi, ())

    @partial(jit, static_argnums=(0,))
    def R_nl_sp(self, r_i, R_n0_params, R_n1_params):
        """ Single-particle radial functions 
        """
        rmod_i = jnp.sqrt(jnp.sum(r_i ** 2))
        R_n0_i = jnp.exp(self.R_n0_apply(R_n0_params, rmod_i))
        R_n1_i = jnp.exp(self.R_n1_apply(R_n1_params, rmod_i))

        #        print("rmod_i=", rmod_i)
        #        print("R_n0_i=", R_n0_i[0][0])
        #        print("R_n1_i=", R_n1_i[0][0])

        #        print("R_nl_i_1=", R_nl_i[0][1])
        R_nl = jnp.zeros(2)
        R_nl = index_update(R_nl, 0, R_n0_i[0][0] * jnp.exp(- self.conf * rmod_i ** 2))
        R_nl = index_update(R_nl, 1, R_n1_i[0][0] * jnp.exp(- self.conf * rmod_i ** 2))
        #        exit()

        return R_nl

    @partial(jit, static_argnums=(0,))
    def Y_1m_sp(self, r_i):
        norm = jnp.sqrt(3 / 4 / jnp.pi)
        Y_1m = jnp.zeros(3)
        Y_1m = index_update(Y_1m, 0, norm * r_i[1])
        Y_1m = index_update(Y_1m, 1, norm * r_i[2])
        Y_1m = index_update(Y_1m, 2, norm * r_i[0])
        return Y_1m

    @partial(jit, static_argnums=(0,))
    def vmap_psi(self, params, r_batched, sz_batched):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    def vmap_psi_sum(self, params, r_batched):
        return jnp.sum(self.vmap_psi(params, r_batched))

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters

    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters

    @partial(jit, static_argnums=(0,))
    def update_add(self, params, dparams):
        return params + dparams

    @partial(jit, static_argnums=(0,))
    def update_subtract(self, params, dparams):
        return params - dparams

    @partial(jit, static_argnums=(0,))
    def update_mix(self, params, dparams):
        return self.mix * params + (1 - self.mix) * dparams

    @partial(jit, static_argnums=(0,))
    def update_cast(self, params):
        return params.astype(jnp.float64)

    @partial(jit, static_argnums=(0,))
    def update_zero(self, params):
        return 0 * params
