def get_dr_ij(r_coords, particle_pairs):
    """

    Parameters
    ----------
    r_coords: ndarray [n_particles, n_coords]
    particle_pairs: ndarray [n_pairs, 2]

    Returns
    -------
    \\delta r for each particle pair

    """
    return r_coords[particle_pairs[:, 0]] - r_coords[particle_pairs[:, 1]]
