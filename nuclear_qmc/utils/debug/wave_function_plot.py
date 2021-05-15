import numpy as np
from matplotlib import pyplot as plt


def get_wave_function_values_per_r_ij(psi_scalar_function, params, n_particles, n_dimensions, particle_pairs, r_ij_max,
                                      step_size=0.05):
    n_points = int(r_ij_max // step_size)
    r_ijs = np.arange(0, r_ij_max, step_size)
    values = dict()
    for p1, p2 in particle_pairs:
        r_coords = np.zeros(shape=(n_points + 1, n_particles, n_dimensions))
        r_coords[:, p2, 0] = r_ijs
        psi_r = np.array([psi_scalar_function(params, r) for r in r_coords])
        name = f'r_{p1}{p2}'
        values[name] = [r_ijs, psi_r]
    return values


def get_wave_function_plot(psi_scalar_function, params, n_particles, n_dimensions, particle_pairs, r_ij_max,
                           step_size=0.05):
    values = get_wave_function_values_per_r_ij(psi_scalar_function, params, n_particles, n_dimensions, particle_pairs,
                                               r_ij_max, step_size=step_size)
    names = list(values.keys())
    for i, name in enumerate(names):
        r_ij, current_values = values[name]
        plt.plot(r_ij, current_values, label=name)
    plt.legend()
    return plt
