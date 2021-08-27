from nuclear_qmc.operators.hamiltonian.get_local_energy import get_local_energy
import numpy as np
from scipy.stats import gaussian_kde
import jax.numpy as jnp
from jax import vmap
from matplotlib import pyplot as plt


def get_average_distance_from_center(r_coords):
    out = jnp.linalg.norm(r_coords, axis=1)
    out = out.mean()
    return out


def plot_local_energy(psi_prefactor, psi_params, psi_vector, hamiltonian, block_samples, plot_title_str):
    # get local energy
    local_energy_values = jnp.array([vmap(get_local_energy, in_axes=(None, None, None, 0, None))(psi_prefactor
                                                                                                 , psi_params
                                                                                                 , psi_vector
                                                                                                 , samples
                                                                                                 , hamiltonian)
                                     for samples in block_samples])
    local_energy_values = local_energy_values.reshape(-1)

    # get average distance from center
    average_distance_from_center = jnp.array(
        [vmap(get_average_distance_from_center)(samples) for samples in block_samples])
    average_distance_from_center = average_distance_from_center.reshape(-1)

    # plot
    plt.clf()

    plt.scatter(average_distance_from_center, local_energy_values, label='local energy')

    # sample density
    density = gaussian_kde(average_distance_from_center)
    mn = average_distance_from_center.min()
    mx = average_distance_from_center.max()
    dx = 0.1
    numx = int(abs(mx - mn) / dx)
    density.covariance_factor = lambda: .25
    xs = np.linspace(mn, mx, numx)
    density._compute_covariance()
    y = density(xs)
    y_range = y.max() - y.min()
    y *= 0.8 * (local_energy_values.max() - local_energy_values.min()) / y_range
    y -= y.min() - local_energy_values.min()
    plt.plot(xs, y, label='sample density', ls='--')

    # plot average local energy
    min_x = average_distance_from_center.min()
    max_x = average_distance_from_center.max()
    mean_local = local_energy_values.mean()
    plt.hlines(mean_local, min_x, max_x, colors='r', label='average local energy')
    plt.hlines(mean_local + local_energy_values.std(), min_x, max_x, colors='r', ls='--')
    plt.hlines(mean_local - local_energy_values.std(), min_x, max_x, colors='r', ls='--')
    plt.text(min_x + 0.05 * (max_x - min_x), mean_local + 1.03, f'{mean_local.round(3)}')

    plt.xlabel('Average Distance From Center [fm]')
    plt.ylabel('Energy [MeV]')

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(plot_title_str, bbox_extra_artists=(lgd,), bbox_inches='tight')
