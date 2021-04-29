import jax.numpy as jnp
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

    def sigma(self, pair_coefficients):
        return self._tau_or_sigma(self.spin_exchange_indices, pair_coefficients)

    def tau(self, pair_coefficients):
        return self._tau_or_sigma(self.isospin_exchange_indices, pair_coefficients)

    def _tau_or_sigma(self, exchange_indices, pair_coefficients):
        sigma_spin = self.spin[exchange_indices]
        sigma_spin = 2.0 * sigma_spin - self.spin.reshape(-1, 1)
        sigma_spin *= pair_coefficients
        return sigma_spin.sum(axis=1)

    def weight(self, r_coords):
        psi = self.psi(r_coords)
        return jnp.vdot(psi, psi)

    @abstractmethod
    def psi(self, r_coords):
        return self.spin
