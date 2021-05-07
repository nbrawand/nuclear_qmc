import jax.numpy as jnp


def psi_prefactor(params, r_coords):
    rcm = jnp.mean(r_coords, axis=0)
    r = r_coords - rcm[None, :]
    delta_r = jnp.linalg.norm(r[1, :] - r[0, :])
    return jnp.exp(- delta_r / params[0])
