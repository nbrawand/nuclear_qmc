import argparse
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pymc3 as pm


def fit_exp_model(energy_values, final_energy_estimate_name, sample_size):
    with pm.Model() as model:
        """
        energy ~ Normal(mu, sigma)
        mu ~ a*exp(-b*opt_step) + c
        a ~ HalfCauchy
        b ~ HalfCauchy
        c ~ Gaussian
        """
        sigma = pm.distributions.HalfCauchy("sigma", beta=10)
        a = pm.HalfCauchy("a", beta=10)
        b = pm.HalfCauchy("b", beta=10)
        c = pm.Normal(final_energy_estimate_name, mu=np.mean(energy_values[-10:]), sigma=1)
        opt_step = range(len(energy_values))
        mu = a * np.exp(-b * opt_step) + c
        likelihood = pm.Normal("energy", mu=mu, sigma=sigma, observed=energy_values)

        trace = pm.sample(sample_size)  # draw 3000 posterior samples using NUTS sampling
    map_estimate = pm.find_MAP(model=model)
    return trace, map_estimate


def get_energy_values(lines):
    search_str = 'optimization step'
    opt_lines = [l for l in lines if search_str in l[:len(search_str)]]  # get lines that start with search string
    energy_values = [l.split('|')[2] for l in opt_lines]  # split by the pipe character and get the 3rd element (energy)
    energy_values = np.array(energy_values, dtype=np.float64)
    return energy_values


def get_optimization_plot(markdown_file, start):
    start = int(start)
    with open(markdown_file, 'r') as fil:
        lines = fil.readlines()
    energies = get_energy_values(lines)
    energies = energies[start:]
    plt.plot(energies, 'b.')
    plt.savefig('energy')


def get_results(function_name, func_args):
    return eval(function_name)(*func_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='post-processing routines for nuclear qmc runs.')
    parser.add_argument("-c", dest="command", type=str, help="command name to compute result")
    parser.add_argument("-a", dest="func_args", type=str, help="function args", nargs='+')
    args = parser.parse_args()
    run_results = get_results(args.command, args.func_args)
