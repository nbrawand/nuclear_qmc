import jax.numpy as jnp
from nuclear_qmc.wave_function.wave_function import get_wave_function_system

from nuclear_qmc.constants.constants import H_BAR_SQRD_OVER_2_M
from nuclear_qmc.operators.operators import sigma_psi_r, kinetic_energy_psi, tau_psi_r, sigma_tau_psi_r


class TestOperators:
    def test_sigma_r(self):
        wfc = jnp.array([[0., 1., 2., 3], [4., 5, 6, 7], [8, 9, 10, 11]])  # 4 spin states, 3 isospin states
        xi = jnp.array([[0, 2, 1, 3], [3, 1, 2, 0]]).T  # 2 pair exchanges
        pair_coefs = jnp.array([0, 1])
        expected = jnp.array([[6., 1., 2., -3.],
                              [10., 5., 6., 1.],
                              [14., 9., 10., 5.]])
        computed = sigma_psi_r(wfc, xi, pair_coefs)
        assert jnp.array_equal(expected, computed)

    def test_sigma_r_single_exchange(self):
        wfc = jnp.array([[0., 1., 2., 3], [4., 5, 6, 7], [8, 9, 10, 11]])  # 4 spin states, 3 isospin states
        xi = jnp.array([[0, 2, 1, 3]]).T  # 1 pair exchange
        pair_coefs = 1
        expected = jnp.array([[0., 3., 0., 3.],
                              [4., 7., 4., 7.],
                              [8., 11., 8., 11.]])
        computed = sigma_psi_r(wfc, xi, pair_coefs)
        assert jnp.array_equal(expected, computed)

    def test_kinetic_energy_base_wfc(self):
        spin = jnp.array([1, 0])
        params = jnp.array([1.])
        psi = lambda p, x: x.reshape(-1).sum() ** 2
        r_coords = jnp.array([[1., 0., 0.], [0., 0., 0.]])
        grad_psi = 2.0 * spin
        n_particles = 2
        n_dims = 3
        ke_psi = - H_BAR_SQRD_OVER_2_M * grad_psi * n_particles * n_dims
        psi_r = psi(r_coords, params) * spin
        expected = jnp.dot(psi_r, ke_psi)
        computed = kinetic_energy_psi(psi, params, r_coords) * spin
        computed = jnp.dot(psi_r, computed)
        assert expected == computed

    def test_tau_r(self):
        particle_pairs, particle_triplets, spin, spin_exchange_indices, isospin_exchange_indices = get_wave_function_system(
            1, 2)
        pairs = jnp.array([0, 1, 2])
        computed = tau_psi_r(spin, isospin_exchange_indices, pairs)
        expected = jnp.array([[0., -2., 1., 0., 1., 0., 0., 0.],
                              [0., -5., 4., 0., 1., 0., 0., 0.],
                              [0., 7., -5., 0., -2., 0., 0., 0.]], dtype=jnp.float64)
        assert jnp.array_equal(expected, computed)

    def test_sigma_tau_psi_r(self):
        particle_pairs, particle_triplets, spin, spin_xi, iso_xi = get_wave_function_system(
            1, 2)
        pairs = jnp.array([0, 1, 2])
        computed = sigma_tau_psi_r(spin, spin_xi, iso_xi, pairs)
        tau_ij = jnp.array([tau_psi_r(spin, iso_xi[:, i], pairs[i]) for i in range(3)])
        expected = jnp.array([sigma_psi_r(tau_ij[i], spin_xi[:, i].reshape(-1, 1), 1) for i in range(3)]).sum(axis=0)
        assert jnp.array_equal(computed, expected)
