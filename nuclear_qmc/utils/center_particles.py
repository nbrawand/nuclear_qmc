import jax.numpy as jnp


def center_particles(r_coords):
    """Return r_coords with center of mass removed.

    Parameters
    ----------
    r_coords: ndarray
        [n_particles, n_dimensions] particle coordinates.

    Returns
    -------
    r_coords: ndarray
        Particle coordinates with center of mass removed.

    """
    rcm = jnp.mean(r_coords, axis=0)
    r_coords = r_coords - rcm[None, :]
    return r_coords
