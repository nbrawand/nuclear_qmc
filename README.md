![Tests Status](https://github.com/nbrawand/nuclear_qmc/actions/workflows/main.yml/badge.svg)

# Nuclear QMC
This package contains tools for running Quantum Monte Carlo simulations of nuclei. The code leverages GPU resources 
to accelerate computations. 

## Installation
The code is tested using python versions 3.7, 3.8, and 3.9.
### CPU
For basic CPU installs, code dependencies can be installed using the [requirements file](/requirements.txt) with the following command:
```
pip install -r requirements.txt
```
### GPU
Please follow the jax installation instructions on the jax website if GPUs are necessary.

## Tests & Github Workflows
Testing is done using the pytest package. All tests are in the [tests directory](/tests). Tests can be run manually using:
```
python -m pytest
```
Pytests are automatically run for each push to the main branch using github workflows controlled by
[.github/workflows/main.yml](/.github/workflows/main.yml).

## Examples
Examples for using the code can be found in the [runs directory](/runs).

## Further Documentation
Additional documentation can be found in the [documentation directory](/documentation).
* [input specification](/documentation/input.md)
* [wave function conventions and spin-isospin operators](/documentation/wave_function_conventions.md)
* [wave function antisymmetry](/documentation/wave_function_antisymmetry.md)


[comment]: <> (this is how u do math in markdown: <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)
