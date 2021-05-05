from jax.config import config
import jax.numpy as jnp
import jax
from jax.experimental import optimizers
from nuclear_qmc.operators.operators import kinetic_energy_psi
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
from nuclear_qmc.wave_function.test_neural_network import NeuralNetworkTestWaveFunction as WaveFunction
from nuclear_qmc.sampling.sample import sample, center_walkers

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 0.3
WALKER_STEP_SIZE = 1.0
N_WALKERS = 4000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 100
N_STEPS = 8
N_VOID_STEPS = 200
N_OPTIMIZATION_STEPS = 2000
LEARNING_RATE = 0.001

wave_function = WaveFunction()
particle_pairs = wave_function.particle_pairs
particle_triplets = wave_function.particle_triplets
spin_exchange_indices = wave_function.spin_exchange_indices
spin = wave_function.spin

KEY = random.PRNGKey(SEED)
global KEY

KEY, key_input = jax.random.split(KEY)
x_o = INITIAL_WALKER_STANDARD_DEVIATION * jax.random.normal(key_input,
                                                            shape=[N_WALKERS, N_PROTON + N_NEUTRON, N_DIMENSIONS],
                                                            dtype=jnp.float64)
global x_o

x_o = center_walkers(x_o)


def loss_fn(params):
    global KEY, x_o

    def psi(r_coords):
        vec = wave_function.psi_vector(r_coords, params, spin)
        prefactor = wave_function.psi_prefactor(r_coords, params)
        return prefactor * vec

    def weight_function(r_coords):
        prefactor = wave_function.psi_prefactor(r_coords, params)
        return prefactor

    KEY, r_coord_samples = sample(
        weight_function
        , N_STEPS
        , INITIAL_WALKER_STANDARD_DEVIATION
        , WALKER_STEP_SIZE
        , N_WALKERS
        , N_NEUTRON + N_PROTON
        , N_DIMENSIONS
        , N_EQUILIBRIUM_STEPS
        , N_VOID_STEPS
        , KEY
        , x_o
    )
   
    x_o = r_coord_samples[-1]

    r_coord_samples = r_coord_samples.reshape(-1, N_PROTON + N_NEUTRON, N_DIMENSIONS)
    local_energy = vmap(get_local_energy, in_axes=(None, 0, None, None, None, None))(psi
                                                                                     , r_coord_samples
                                                                                     , particle_pairs
                                                                                     , particle_triplets
                                                                                     , spin_exchange_indices
                                                                                     , kinetic_energy_psi).mean()
    return local_energy


opt_init, opt_update, get_params = optimizers.adagrad(LEARNING_RATE)
opt_state = opt_init(wave_function.params)


def step(step, opt_state):
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    grads.block_until_ready()
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


for i in range(N_OPTIMIZATION_STEPS):
    value, opt_state = step(i, opt_state)
    print(i, value)

wave_function.params = get_params(opt_state)
wave_function.save('wfc.model')
