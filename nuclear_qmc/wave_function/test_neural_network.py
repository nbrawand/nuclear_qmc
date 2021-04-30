import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.experimental import stax
from jax.experimental.stax import Dense, elementwise, Tanh
from functools import partial
import pickle
from nuclear_qmc.wave_function.wave_function import WaveFunction
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


@jit
def lintanh(x):
    return x - jnp.tanh(x) / 2


Lintanh = elementwise(lintanh)
Sin = elementwise(jnp.sin)


class TestNeuralNetwork(WaveFunction):
    def __init__(self, params_file=os.path.join(dir_path, 'test_neural_network.model')):
        super().__init__(n_protons=1, n_neutrons=1)
        self.ndim = 3
        self.npart = self.n_protons + self.n_neutrons
        self.conf = 0.1
        self.key = random.PRNGKey(0)
        self.mix = 0.0
        self.ndense = 8
        self.nlat = 32  ## 1 * (self.ndim * self.npart + 4)
        self.activation = Tanh
        self.a = 8
        self._build()
        self.params_file = params_file
        if self.params_file is not None:
            self.params = self.load_params(self.params_file)

    def _build(self):
        # phi_a
        self.phi_a_init, self.phi_a_apply = stax.serial(
            Dense(self.ndense), self.activation,
            Dense(self.ndense), Tanh,
            Dense(1),
        )
        in_shape = (1,)
        self.key, key_input = jax.random.split(self.key)
        phi_a_shape, phi_a_params = self.phi_a_init(key_input, in_shape)
        self.num_phi_a_params = len(phi_a_params)
        self.params = phi_a_params

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.params, file)

    def load_params(self, params_file_name):
        with open(params_file_name, 'rb') as fil:
            return pickle.load(fil)

    def weight(self, r):
        """The weight that was used in the original code used psi(r) not |psi(r)|^2 so this gives the correct sampling stats"""
        return self._psi_r(r)

    def _psi_r(self, r):
        rcm = jnp.mean(r, axis=0)
        r = r - rcm[None, :]
        delta_r = jnp.linalg.norm(r[0, :] - r[1, :])

        phi_a_params = self.params[0:self.num_phi_a_params]
        phi_a_out = self.phi_a_apply(phi_a_params, delta_r)
        phi_a_out = jnp.mean(phi_a_out)

        psi = jnp.exp(phi_a_out)
        psi *= self.phi(r)
        return jnp.reshape(psi, ())

    @partial(jit, static_argnums=(0,))
    def psi(self, r):
        return self._psi_r(r) * self.spin

    @partial(jit, static_argnums=(0,))
    def phi(self, r):
        """ Boundary condition imposed on multiple particles
        """
        rcm = jnp.mean(r, axis=0)
        r = r - rcm[None, :]
        return jnp.prod(vmap(self.sp_boundary, in_axes=(0))(r))

    @partial(jit, static_argnums=(0,))
    def sp_boundary(self, r):
        """ Boundary condition imposed on single particle 
        """
        sp_conf = jnp.exp(- self.conf * jnp.sum(r ** 2))

        return sp_conf
