from nuclear_qmc.wave_function.jastro_neural_network_builder.build_nn_wave_function import build_nn_wave_function
import jax.numpy as jnp


def get_nn_jastro_func_and_params(key
                                  , n_dense
                                  , n_hidden_layers
                                  , jastro_builder
                                  , jastro_builder_args
                                  , nn_wrapper_function=jnp.exp
                                  ):
    """Build jastro function using a neural network

    Parameters
    ----------
    key: jax key
    n_dense: int
        number of neurons
    n_hidden_layers: int
        number of hidden layers
    jastro_builder: function
        The jastro builder function
    jastro_builder_args: list
        List of argumments to be added to the jastro builder function after the nn function is passed in
    nn_wrapper_function
        The function that wraps the neural network output. Common options are exp or tanh.

    Returns
    -------
    key
    jastro_func
        The jastro function parameterized by a neural network.
    params
        The neural network parameters.

    """
    key, nn, params = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)
    func = lambda p, r_ij: nn_wrapper_function(nn(p, r_ij))
    jastro_func = jastro_builder(func, *jastro_builder_args)
    return key, jastro_func, params
