![Tests Status](https://github.com/nbrawand/nuclear_qmc/actions/workflows/main.yml/badge.svg)

# Nuclear QMC
This package contains tools for running Quantum Monte Carlo simulations of nuclei. The code leverages GPU resources 
to accelerate computations. The project is in its infancy and interfaces are subject to significant change.

## Installation
The code base is not a package yet (this will change in the future) but can be used using the following:
```
export PYTHONPATH="$PWD"
pip install -r requirements.txt
```
The package jax is necessary to install and run please see the jax website for installation instructions.

## Tests
Testing is done using the pytest package. All tests are in the [tests directory](/tests). Tests can be run using:
```
python -m pytest
```

## Examples
Examples for using the code can be found in the [runs directory](/runs).

