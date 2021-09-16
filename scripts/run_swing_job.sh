#!/bin/bash
#SBATCH --job-name=6li
#SBATCH --account=NNQMC
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/home/nbrawand/nuclear_qmc
srun python /home/nbrawand/nuclear_qmc/main.py -i input.json
