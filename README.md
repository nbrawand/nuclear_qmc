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

## Running The Code and Examples
### Adding nuclear qmc to your path
The code can be run from any directory but you will need to make sure that the package "nuclear_qmc",
under the root directory, is in your python path. You can add it by navigating to the root directory and
executing the command:
~~~
export PYTHONPATH=/<path>/nuclear_qmc
~~~
### Running the code
To run the code execute the following command:
~~~
python /<path>/nuclear_qmc/main.py -i /<path>/input.json
~~~
The option -i indicates that the following argument will be read as the input file.
See the documentation section about input format.
Examples for using the code can be found in the [runs directory](/runs).
A job script for swing can be found [here](/scripts/run_swing_job.sh).

## Further Documentation
Additional documentation can be found in the [documentation directory](/documentation).
* [input specification](/documentation/input.md)
* [wave function conventions and spin-isospin operators](/documentation/wave_function_conventions.md)


[comment]: <> (this is how u do math in markdown: <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)
