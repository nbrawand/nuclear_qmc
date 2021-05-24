import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import os

from post_processing.post_processing import get_energy_values, fit_exp_model

scrpt_dr = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(scrpt_dr, 'nuclear_qmc.md'), 'r') as fil:
    lines = fil.readlines()
energy_values = get_energy_values(lines)
energy_values = energy_values[500:]
trace, map_est = fit_exp_model(energy_values, final_energy_estimate_name='final_energy_estimate', sample_size=4000)
az.plot_posterior(trace, var_names='final_energy_estimate', round_to=4)
plt.title('3H 3Body')
plt.savefig(os.path.join(scrpt_dr, 'final_energy_estimate'))
plt.clf()
x = range(len(energy_values))
plt.scatter(x, energy_values, alpha=0.85, label='data')
fit = [map_est['a'] * np.exp(-map_est['b'] * xx) + map_est['final_energy_estimate'] for xx in x]
plt.plot(x, fit, label='A*exp(-b step)+final_energy_estimate', c='red')
plt.legend()
plt.title('3H 3Body')
plt.savefig(os.path.join(scrpt_dr, 'opt_plot'))
