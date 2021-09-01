from nuclear_qmc.wave_function.wave_function_builder.build_wave_function import build_wave_function, get_states
import numpy as np
from jax import random
import jax.numpy as jnp
from time import time


class TestGetWFC:
    # def test_build_wave_function_6Li_speed(self):
    #     key = random.PRNGKey(0)
    #     states = [
    #         [
    #             "R0_Y00_d_n",
    #             "R0_Y00_u_n",
    #             "R1_Y11_d_n",
    #             "R0_Y00_d_p",
    #             "R0_Y00_u_p",
    #             "R1_Y11_d_p"
    #         ],
    #         [
    #             "R0_Y00_d_n",
    #             "R0_Y00_u_n",
    #             "R1_Y10_d_n",
    #             "R0_Y00_d_p",
    #             "R0_Y00_u_p",
    #             "R1_Y10_d_p"
    #         ],
    #         [
    #             "R0_Y00_d_n",
    #             "R0_Y00_u_n",
    #             "R1_Y1m1_d_n",
    #             "R0_Y00_d_p",
    #             "R0_Y00_u_p",
    #             "R1_Y1m1_d_p"
    #         ]
    #     ]
    #     key, psi, params = build_wave_function(key
    #                                            , n_neutron=3
    #                                            , n_proton=3
    #                                            , n_dense=1
    #                                            , n_hidden_layers=1
    #                                            , states=states
    #                                            , coefficients=jnp.array([1., 1., 1.])
    #                                            , add_partition_jastro=True
    #                                            , confining_factor=0.0
    #                                            )
    #     r = jnp.array(
    #         [[0.5977331, 0.25465614, 0.50383615],
    #          [0.964289, 0.21575487, 0.20640086],
    #          [0.7600208
    #              , 0.59024197
    #              , 0.79093385],
    #          [0.38367015, 0.2965703, 0.00292212],
    #          [0.5977433
    #              , 0.93917906
    #              , 0.8908537],
    #          [0.2506502, 0.7162357, 0.8377191]]
    #     )
    #     times = []
    #     computed = psi(params, r)
    #     for t in range(10):
    #         t1 = time()
    #         computed2 = psi(params, r)
    #         computed2.block_until_ready()
    #         times.append(time() - t1)
    #     print('Average time for wfc execution:', np.mean(times))
    #     # jnp.save('li6_wave_function_for_testing.npy', computed.block_until_ready())
    #     expected = jnp.array(jnp.load('li6_wave_function_for_testing.npy'))
    #     assert jnp.array_equal(computed.round(4), expected.round(4))

    def test_build_wave_function_2H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=1
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2
                                               , states=[['1_1_d_n', '1_1_d_p']]
                                               , coefficients=jnp.array([1.])
                                               , confining_factor=0.0
                                               )
        r = jnp.ones(shape=(2, 3))
        computed = psi(params, r)
        expected = jnp.array([[1., 0., 0., 0.],
                              [-1., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)

    def test_build_wave_function_3H(self):
        key = random.PRNGKey(0)
        key, psi, params = build_wave_function(key
                                               , n_neutron=2
                                               , n_proton=1
                                               , n_dense=2
                                               , n_hidden_layers=2
                                               , states=[['1_1_d_n', '1_1_u_n', '1_1_d_p']]
                                               , coefficients=jnp.array([1.])
                                               , confining_factor=0.0
                                               )
        r = jnp.ones(shape=(3, 3))
        computed = psi(params, r)
        expected = jnp.array([[0., 0., 1., 0., -1., 0., 0., 0.],
                              [0., -1., 0., 0., 1., 0., 0., 0.],
                              [0., 1., -1., 0., 0., 0., 0., 0.]])
        assert jnp.array_equal(computed, expected)


def test_get_states():
    c = get_states(n_protons=1, n_neutrons=1)
    e = ['Sdn', 'Sdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=1)
    e = ['Sdn', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=1, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee


def test_get_states_pshell():
    c = get_states(n_protons=3, n_neutrons=2)
    e = ['Sdn', 'Sun', 'Sdp', 'Sup', 'Pdp']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=2, n_neutrons=3)
    e = ['Sdn', 'Sun', 'Pdn', 'Sdp', 'Sup']
    for cc, ee in zip(c, e):
        assert cc == ee

    c = get_states(n_protons=3, n_neutrons=3)
    e = ['Sdn', 'Sun', 'Pdn', 'Sdp', 'Sup', 'Pdp']
    for cc, ee in zip(c, e):
        assert cc == ee
