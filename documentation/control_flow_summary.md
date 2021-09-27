# Control flow summary
This is a summary of the execution order of the main routine.

## Argparse and logging 
The main routine [main.py](../main.py) loads a jason input file into a python dictionary. Default input values are
added to the dictionary if not present in the input file. Parameter expansion is used in some cases to call functions;
when this happens the functions will dictate default arguments.  The code creates a logfile nuclear_qmc.md in
the directory where the main routine is called. If a log file already exists, it is appended to. The log file is in 
markdown format  which can be rendered and will contain a copy of the input which can be used as 
input to separate calculations for reproducibility.

## System arrays
Various items such as the hamiltonian and wave function rely on arrays such as particle pair indices and
spin exchange indices. These arrays are computed once at the start of the calculation and 
referenced by other subroutines.

## Wave function
### construction
The wave function is constructed in 2 stages first the determinants specified by the lists in the input file are
constructed by [build_wave_function](https://github.com/nbrawand/nuclear_qmc/blob/694f30fd01d8a7645518febf9a9e7a0205b890fc/nuclear_qmc/wave_function/wave_function_builder/build_wave_function.py#L220).
This routine also constructs the partition jastro discussed in a different readme. The function is responsible for
evaluating the coordinates in each single particle orbital, constructing the different elements of the GFMC wave function,
adding up the separate determinants and returning the resulting wave function.
The 2nd stage of the wave function construction if building the other jastro factors and multiplying them with the wave
function built in the previous stage.

### parameters
The main routine will search and read parameters from a file if specified in the input. If a file is specified but
not found, random parameters will be initialized and saved to the specified file. If not specified params will be 
initialized and written to wave_function_parameters_0.npy in the execution directory.

## Hamiltonian & operators
The potential can be specified by the input file. The main hamiltonian builder function returns H|psi>. The operators 
sigma and tau utilize the spin and isospin arrays. Two options exist for computing the derivatives
in the kinetic energy: autodiff and finite diff. The autodiff requires significantly more memory and doesn't appear
to be any faster than finite diff. One benefit is that the autodiff is more accurate than the finite diff method 
implemented.

## Sampling & Optimization
New samples are drawn from the wave function using the metropolis algorithm. The weight function used is:
<img src="https://render.githubusercontent.com/render/math?math=\sum_i |\Psi_i|^2">
where the sum is over the spin-isospin configurations. n_blocks*n_equilibrium_steps are taken before saving walkers.
 The number of samples drawn is given by n_blocks*n_walkers with n_void_steps taken between each walker. The SR equations
 are solved after the samples are collected and the updates are added to the wave function parameters. 
