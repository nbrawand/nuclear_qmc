import jax.numpy as jnp
from jax import jit
from abc import abstractmethod
from nuclear_qmc.spin.get_spin_isospin_wave_function import get_spin_isospin_wave_function
from nuclear_qmc.spin.get_tables import get_spin_particle_pairs, get_spin_exchange_indices, get_isospin_exchange_index, \
    get_spin_state_indices
from nuclear_qmc.utils.get_triplets import get_triplets


class WaveFunction:

    def __init__(self, n_protons, n_neutrons, include_isospin=True):
        self.n_protons = n_protons
        self.n_neutrons = n_neutrons
        self.include_iso_spin = include_isospin
        self._initialize_spin_isospin()
        self.spin = get_spin_isospin_wave_function(self.n_protons, self.n_neutrons, include_isospin=include_isospin)

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
        if not psi_r:
            psi_r = self.psi(r_coords)
        return self._tau_or_sigma(psi_r, self.spin_exchange_indices, pair_coefficients)

    def tau(self, r_coords, pair_coefficients, psi_r=None):
        if not psi_r:
            psi_r = self.psi(r_coords)
        return self._tau_or_sigma(psi_r, self.isospin_exchange_indices, pair_coefficients)

    @staticmethod
    @jit
    def _tau_or_sigma(psi_r, exchange_indices, pair_coefficients):
        sigma_psi = psi_r[exchange_indices]
        sigma_psi = 2.0 * sigma_psi - psi_r.reshape(-1, 1)
        sigma_psi *= pair_coefficients
        return sigma_psi.sum(axis=1)

    @abstractmethod
    def psi(self, r_coords):
        return self.spin
