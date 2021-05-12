import jax.numpy as jnp


def center_particles(r_coords):
    """
    return r_coords with center of mass removed
    Parameters
    ----------
    r_coords: ndarray [n_particles, n_dimensions]

    Returns
    -------

    """
    rcm = jnp.mean(r_coords, axis=0)
    r_coords = r_coords - rcm[None, :]
    return r_coords
