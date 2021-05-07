import jax.numpy as jnp


def psi_prefactor(params, r_coords):
    rcm = jnp.mean(r_coords, axis=0)
    r = r_coords - rcm[None, :]
    delta_r1 = jnp.linalg.norm(r[0, :]) ** 2
    delta_r2 = jnp.linalg.norm(r[1, :]) ** 2
    return jnp.exp(- (delta_r1 + delta_r2) / params[0] ** 2)
