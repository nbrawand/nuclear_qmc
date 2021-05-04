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

    @property
    def params(self):
        if not hasattr(self, '_params'):
            self._params = jnp.array([1.])
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def psi(self, r_coords):
        spin = self.psi_vector(r_coords, self.params, self.spin)
        prefactor = self.psi_prefactor(r_coords, self.params)
        return prefactor * spin

    def psi_prefactor(self, r_coords, params):
        psi_prefac = jnp.sum(r_coords ** 2, axis=-1)
        psi_prefac = jnp.sum(psi_prefac)
        psi_prefac = jnp.exp(-params[0] * psi_prefac)
        return psi_prefac

    def psi_vector(self, r_coords, params, spin):
        return spin
