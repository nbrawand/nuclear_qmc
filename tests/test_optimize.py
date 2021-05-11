from nuclear_qmc.optimize.optimize import get_delta_params
import jax.numpy as jnp


class TestOptimize:
    def test_get_new_wave_function_parameters(self):
        """A loss function that is the diff of psi and a parabola is optimized for several steps.
        The parameter should match the parabola center"""
        offset = abs(1.0)
        psi = lambda p, x: (x - p[0]) ** 2 + offset  # the offset is needed to avoid zero.
        target_parameter_value = 1.5
        psi_params = jnp.array([0.8])
        psi_vector = jnp.array([1., 1.])
        r_coords = jnp.arange(-10, 10.01, 0.01)
        particle_pairs = None
        particle_triplets = None
        spin_exchange_indices = None
        learning_rate = 0.1

        def hamiltonian(psi
                        , psi_params
                        , psi_vector
                        , r_coords
                        , particle_pairs
                        , particle_triplets
                        , spin_exchange_indices):
            return -((r_coords - target_parameter_value) ** 2 - psi(psi_params, r_coords)) ** 2 * psi_vector

        for n in range(20):
            delta_params, loss = get_delta_params(
                psi
                , psi_params
                , psi_vector
                , r_coords
                , particle_pairs
                , particle_triplets
                , spin_exchange_indices
                , learning_rate
                , hamiltonian=hamiltonian
                , include_sr_equations=False
                , return_loss=True)
            psi_params += delta_params
        computed = psi_params.round(1)
        assert target_parameter_value == computed + offset
