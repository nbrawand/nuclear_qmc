# Input Specification
Input is specified using the json standard. See run directory for input examples.
Here is and example for 2H:
```json
{
  "wave_function": {
    "confining_factor": 0.01,
    "add_partition_jastro": true,
    "seed": 0,
    "n_dense": 4,
    "n_hidden_layers": 1,
    "jastro_list": [
      "2b",
    ],
    "wave_function_file": "wave_function_parameters_0.npy",
    "coefficients": [1.0],
    "orbitals": [["R0_Y00_d_n", "R0_Y00_d_p"]]
  },
  "potential_energy": "arxiv_2102_02327v1",
  "potential_kwargs": {
    "model_string": "o",
    "R3": 1.0,
    "include_3body":false
  },
  "optimization": {
    "n_dimensions": 3,
    "n_blocks": 4,
    "n_equilibrium_blocks": 4,
    "n_walkers": 200,
    "n_void_steps": 200,
    "walker_step_size": 0.2,
    "initial_walker_standard_deviation": 1.0,
    "n_optimization_steps": 500,
    "learning_rate": 0.001,
    "epsilon_sr": 0.0001,
    "print_local_energy": true,
    "plot_local_energy": false,
    "local_energy_plot_limits": [[0, 16], [-10, 2]],
    "local_energy_plot_start_number": 0
  }
}
```
The json file is read by python and converted to a dictionary. Keys and values of that dictionary are then
used as function arguments to drive the code. Thus the args and kwargs of each function actually determine
what is valid input.

Going through each element of the json file:
~~~
   wave_function - start of the wave function section
     confining_factor - strength of the confining factor applied to the function
     add_partition_jastro - this is a jastro that distinguishes between different partitions between S and P states
     seed - the random seed for the wfc construction
     n_dense - number of neurons per layer when building neural nets
     n_hidden_layers - number of hidden layers when building neural nets
     jastro_list - jastros to apply to the wave function (see below for possible values)
     wave_function_file - file to write to at each optimization step. Is read at start of calculation if file exists.
     coefficients - Coefficients to multiply each determinant in wave function
     orbitals - Single particle orbitals for each determinant. Each lists is a separate determinant.

   potential_energy - Specify the potential energy operator (see below for available options)
   potential_kwargs - start of potential energy kwargs (optional arguments) this is determined by the corresponding function that builds that potential

   optimization - start of the optimization options section
     n_dimensions - dimensionality of system (normally 3d)
     n_blocks - The number of blocks for sampling. Total samples = nblocks x n_walkers
     n_equilibrium_blocks - number of equilibrium blocks before sampling
     n_walkers - number of walkers to sample
     n_void_steps - number of void steps before each walker sample
     walker_step_size  - walker step size for metropolis
     initial_walker_standard_deviation  - std of gaussian for generating walker guesses before metropolis
     n_optimization_steps - total number of optimization steps before ending run
     learning_rate - learning rate in stochastic reconfiguration optimization step
     epsilon_sr  - The small negative number in stochastic reconfiguration optimization step
     print_local_energy - Print local energy at each optimization step to output log 
     plot_local_energy - Save a local energy plot png to disk numbered by optimization step + local_energy_plot_start_number
     local_energy_plot_limits - local energy plot limits [[x1, x2], [y1, y2]]
     local_energy_plot_start_number - integer offset to plot file number
~~~

### Jastro Options
Currently the allowed jastros are determined by the 
[jastro builder routine](https://github.com/nbrawand/nuclear_qmc/blob/d045119bcfc91a10e4883c9aaed49021b090117c/nuclear_qmc/wave_function/neural_network_jastro_builder/add_neural_network_jastros.py#L18).
At this moment, the supported jastros include:
* 2b - two body jastro with argument |r_i - r_j|
* 3b - three body jastro
* sigma - linear sigma_ij jastro
* tau - linear tau_ij jastro
* sigma_tau - linear sigma_ij tau_ij jastro
* add_2b - this is two body jastro where the argument is |r_i + r_j| little utility has been found using this jastro
* add_3b - this is three body jastro where the argument is |r_i + r_j| little utility has been found using this jastro
* pair_deepset - This is a deepset where (r_i, r_j) is used as an argument
* deepset - This is a deepset with r_i as the argument
* total_deepset - This is an experimental feature but little utility has been found using it. 
What it does is outputs multiple values that feed the coefficients for the linear operators.
It is not compatible with the other jastros.

### Nuclear Potential Options
The potential of the hamiltonian is determined by the
[hamiltonian builder function](https://github.com/nbrawand/nuclear_qmc/blob/d045119bcfc91a10e4883c9aaed49021b090117c/nuclear_qmc/operators/hamiltonian/build_hamiltonian.py#L8).
Currently there are 2 potentials to choose from arxiv_2007_14282v2 and arxiv_2102_02327v1. 
The kwargs for these potentials can be viewed at each respective builder function
and specified in input by adding the kwarg to the potential_kwargs section of the input json file.
For example the [latest potential kwargs can be viewed 
here](https://github.com/nbrawand/nuclear_qmc/blob/d045119bcfc91a10e4883c9aaed49021b090117c/nuclear_qmc/operators/hamiltonian/arxiv_2102_02327v1/potential_energy.py#L19). Note that
the user specifies only the kwargs of the potential in the json file directly. In a different branch the latest
potential also includes a "theory_order" kwarg to specify just leading order "lo" or next leading order "nlo" terms.
