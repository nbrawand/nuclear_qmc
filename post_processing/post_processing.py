import argparse
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pymc3 as pm


def exp_model(a, b, c, opt_step):
    return a * np.exp(-b * opt_step) + c


def fit_exp_model(energy_values, final_energy_estimate_name, sample_size):
    trace = None
    with pm.Model() as model:
        """
        energy ~ Normal(mu, sigma)
        mu ~ a*exp(-b*opt_step) + c
        a ~ HalfCauchy
        b ~ HalfCauchy
        c ~ Gaussian
        """
        a = pm.HalfCauchy("a", beta=10)
        b = pm.HalfCauchy("b", beta=1)
        c = pm.Uniform(final_energy_estimate_name, lower=-10.0, upper=0.0)
        # sa = pm.HalfCauchy("sa", beta=0.5)
        # sb = pm.HalfCauchy("sb", beta=10)
        opt_step = np.arange(len(energy_values)) + 1
        beta = pm.Normal('beta', -0.5, 1)
        mu = exp_model(a, b, c, opt_step)  # + opt_step * beta
        sigma = pm.HalfCauchy("sigma", beta=10)  # np.exp(-sa * opt_step) + sb
        likelihood = pm.Normal("energy", mu=mu, sigma=sigma, observed=energy_values)

        if sample_size > 0:
            step = pm.NUTS(adapt_step_size=True, target_accept=0.9)
            trace = pm.sample(sample_size, step=step, tune=20000)  # draw 3000 posterior samples using NUTS sampling
    map_estimate = pm.find_MAP(model=model)
    return trace, map_estimate


def get_energy_values(lines):
    search_str = 'optimization step'
    opt_lines = [l for l in lines if search_str in l[:len(search_str)]]  # get lines that start with search string
    energy_values = [l.split('|')[2] for l in opt_lines]  # split by the pipe character and get the 3rd element (energy)
    energy_values = np.array(energy_values, dtype=np.float64)
    return energy_values


def get_optimization_plot(markdown_file, start, samples, title=None):
    start = int(start)
    with open(markdown_file, 'r') as fil:
        lines = fil.readlines()
    energies = get_energy_values(lines)
    energies = energies[start:]
    plt.plot(energies, 'b.', zorder=0)

    samples = int(samples)
    if samples > 0:
        energy_estimate_name = 'energy_estimate'
        trace, map = fit_exp_model(energies, energy_estimate_name, sample_size=samples)
        x = np.arange(len(energies)) + 1
        exp = exp_model(map['a'], map['b'], map[energy_estimate_name], x)
        plt.plot(exp, 'r-', lw=3, zorder=1)
        plt.hlines(map[energy_estimate_name], 0, len(energies), colors='r', linestyles='--', lw=3, zorder=1,
                   label=f'MAP Estimate: {np.round(map[energy_estimate_name], 4)} MeV')
        plt.legend()

    if title is not None:
        plt.title(title)

    plt.savefig('energy')

    if samples > 0:
        plt.clf()
        az.plot_posterior(trace, var_names=energy_estimate_name)
        plt.savefig('posterior')


def get_results(function_name, func_args):
    return eval(function_name)(*func_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='post-processing routines for nuclear qmc runs.')
    parser.add_argument("-c", dest="command", type=str, help="command name to compute result")
    parser.add_argument("-a", dest="func_args", type=str, help="function args", nargs='+')
    args = parser.parse_args()
    run_results = get_results(args.command, args.func_args)
