![Tests Status](https://github.com/nbrawand/nuclear_qmc/actions/workflows/main.yml/badge.svg)

# Nuclear QMC
This package contains tools for running Quantum Monte Carlo simulations of nuclei. The code leverages GPU resources 
to accelerate computations. 

## Installation
The code is tested using python versions 3.7, 3.8, and 3.9.
### CPU
For basic CPU installs, code dependencies can be installed using the [requirements file](/requirements.txt) with the following command:
```
uv venv -p python3.9  # code written before uv
source .venv/bin/activate
uv pip install -r requirements.txt
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
### Running the code
To run the code execute the following command:
~~~
python /<path>/nuclear_qmc/main.py -i /<path>/input.json
~~~
The option -i indicates that the following argument will be read as the input file.
See the documentation section about input format.
Examples for using the code can be found in the [runs directory](/runs).
A job script for swing can be found [here](/scripts/run_swing_job.sh).

Example:
~~~
python main.py -i runs/h_2/arxiv_2007_14282v2/input.json 
~~~
The output is written to runs/h_2/arxiv_2007_14282v2/nuclear_qmc.md

## Further Documentation
Additional documentation can be found in the [documentation directory](/documentation).
* [Resources including: links, papers, notes, and books](/documentation/resources.md)
* [control flow summary](/documentation/control_flow_summary.md)
* [input specification](/documentation/input.md)
* [wave function conventions and spin-isospin operators](/documentation/wave_function_conventions.md)
* [parallelization](/documentation/parallelization.md)

## To Do
* the arxiv_2102_02327v1 potential has been validated up to 6Li up to LO interactions
* there is an option to include NLO interactions (see input documentation) but only v_c, v_sigma, v_tau, 
and v_sigma_tau parts of NLO
have been unit tested (see test_arxiv_2102_02327v1_potential unit test). 
* The other NLO terms must be validated using the same unit test file and the correct potential 
values found in files: [LO](tests/pot_nn) and [NLO](tests/pot_NLO_nn).

## Contributions
{
  "authors": ["Gnech, A.", "Adams, C.", "Brawand, N.", "et al."],
  "title": "Nuclei with Up to Nucleons with Artificial Neural Network Wave Functions",
  "journal": "Few-Body Systems",
  "volume": 63,
  "article": 7,
  "year": 2022,
  "doi": "10.1007/s00601-021-01706-0",
  "url": "https://doi.org/10.1007/s00601-021-01706-0"
}

[comment]: <> (this is how u do math in markdown: <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)
