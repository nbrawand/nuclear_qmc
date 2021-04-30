import jax.numpy as jnp
import jax
from jax import jit
from abc import abstractmethod

from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M
from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
from nuclear_qmc.spin.get_tables import get_spin_particle_pairs, get_spin_exchange_indices, get_isospin_exchange_index, \
    get_spin_state_indices
from nuclear_qmc.utils.get_triplets import get_triplets


class WaveFunction:

    def __init__(self, n_protons, n_neutrons, include_isospin=True, dtype=jnp.float64):
        self.n_protons = n_protons
        self.n_neutrons = n_neutrons
        self.include_iso_spin = include_isospin
        self._initialize_spin_isospin()
        self.spin = get_spin_isospin_wave_function(self.n_protons, self.n_neutrons
                                                   , include_isospin=include_isospin, dtype=dtype)

    def _initialize_spin_isospin(self):
        mass_number = self.n_protons + self.n_neutrons
        as_jax_array = True
        self.particle_pairs = get_spin_particle_pairs(mass_number, as_jax_array)
        self.particle_triplets = get_triplets(jnp.arange(mass_number))
        self.spin_exchange_indices = get_spin_exchange_indices(self.particle_pairs
                                                               , get_spin_state_indices(mass_number, as_jax_array)
                                                               , as_jax_array)
        if self.include_iso_spin:
            self.isospin_exchange_indices = get_isospin_exchange_index(self.particle_pairs
                                                                       , mass_number
                                                                       , self.n_protons
                                                                       , as_jax_array
                                                                       , also_return_binary_representation=False)

    def sigma(self, r_coords, pair_coefficients, psi_r=None):
        if psi_r is None:
            psi_r = self.psi(r_coords)
        return self._tau_or_sigma(psi_r, self.spin_exchange_indices, pair_coefficients)

    def tau(self, r_coords, pair_coefficients, psi_r=None):
        raise RuntimeError('tau not tested')
        if psi_r is None:
            psi_r = self.psi(r_coords)
        return self._tau_or_sigma(psi_r.T, self.isospin_exchange_indices, pair_coefficients).T

    def kinetic_energy(self, r_coords):
        d2_psi = jax.hessian(self.psi, argnums=0)(r_coords)
        dim = r_coords.shape[0] * r_coords.shape[1]
        d2_psi = d2_psi.reshape(*self.spin.shape, dim, dim)
        d2_psi = jnp.trace(d2_psi, axis1=-1, axis2=-2)
        ke = - H_BAR_SQRD_OVER_2_M * jnp.vdot(self.psi(r_coords), d2_psi)
        return ke

    def weight(self, r_coords):
        psi_r = self.psi(r_coords)
        return jnp.real(jnp.vdot(psi_r, psi_r))

    @staticmethod
    def _tau_or_sigma(psi_r, exchange_indices, pair_coefficients):
        exchanged_psi_r = psi_r[:, exchange_indices]  # for sigma: [n_isospin, n_spin, n_pair_exchanges]
        psi_r = jnp.expand_dims(psi_r, -1)  # for sigma: [n_isospin, n_spin, 1]
        psi_r_prime = 2.0 * exchanged_psi_r - psi_r
        psi_r_prime *= pair_coefficients
        psi_r_prime = psi_r_prime.sum(-1)
        return psi_r_prime

    @abstractmethod
    def psi(self, r_coords):
        psi = jnp.sum(r_coords ** 2, axis=-1)
        psi = jnp.sum(psi)
        psi = jnp.exp(-psi)
        return psi * self.spin
