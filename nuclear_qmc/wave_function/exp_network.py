import jax.numpy as jnp


def psi_prefactor(params, r_coords):
    rcm = jnp.mean(r_coords, axis=0)
    r = r_coords - rcm[None, :]
    n_particles = r.shape[0]
    delta_r = 0
    for i in range(n_particles):
        for j in range(i):
            delta_r += jnp.linalg.norm(r[i, :] - r[j, :])
    return jnp.exp(- delta_r / params[0])
