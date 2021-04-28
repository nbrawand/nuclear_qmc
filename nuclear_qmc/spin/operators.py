from jax import jit, vmap


def sigma(exchange_table, wave_function, pair_coefficients=1):
    sigma_wave_function = wave_function[exchange_table]
    sigma_wave_function = 2.0 * sigma_wave_function - wave_function.reshape(-1, 1)
    sigma_wave_function *= pair_coefficients
    return sigma_wave_function.sum(axis=1)


sigma_gpu = jit(sigma)

batch_sigma_gpu = vmap(sigma_gpu, (None, 0, None))
