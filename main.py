from nuclear_qmc.spin.get_tables import get_tables
from nuclear_qmc.sample import sample
from jax import random

N_PROTON = 2
N_NEUTRON = 2
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 1.0
WALKER_STEP_SIZE = 1.0
N_WALKERS = 100
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 20
N_STEPS = 100
N_VOID_STEPS = 10
WAVE_FUNCTION = None
KEY = random.PRNGKey(SEED)

tables = get_tables(N_PROTON + N_NEUTRON
                    , as_jax_array=True
                    , proton_number=N_PROTON
                    , also_return_binary_representation=False
                    , include_iso_spin=True)

sample(KEY
       , INITIAL_WALKER_STANDARD_DEVIATION
       , WALKER_STEP_SIZE
       , N_WALKERS
       , N_NEUTRON + N_PROTON
       , N_DIMENSIONS
       , N_EQUILIBRIUM_STEPS
       , N_STEPS
       , N_VOID_STEPS
       , WAVE_FUNCTION)
