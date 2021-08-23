#!/bin/bash
#SBATCH --job-name=6he
#SBATCH --account=NNQMC
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/home/nbrawand/nuclear_qmc
srun python /home/nbrawand/nuclear_qmc/main.py -i input.json
