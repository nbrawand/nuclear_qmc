from jax.config import config
import jax.numpy as jnp
import jax
from jax.experimental import optimizers
from nuclear_qmc.operators.operators import kinetic_energy_psi
from jax import random, vmap
from nuclear_qmc.operators.hamiltonian import get_local_energy
from nuclear_qmc.wave_function.exp_network import ExpWaveFunction as WaveFunction
from nuclear_qmc.sampling.sample import sample, center_walkers

config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

N_PROTON = 1
N_NEUTRON = 1
SEED = 0
INITIAL_WALKER_STANDARD_DEVIATION = 0.3
WALKER_STEP_SIZE = 1.0
N_WALKERS = 8000
N_DIMENSIONS = 3
N_EQUILIBRIUM_STEPS = 100
N_STEPS = 8
N_VOID_STEPS = 200
N_OPTIMIZATION_STEPS = 2000
LEARNING_RATE = 0.001

KEY = random.PRNGKey(SEED)

_ = WaveFunction(jnp.array([0.0]))
particle_pairs = _.particle_pairs
particle_triplets = _.particle_triplets
spin_exchange_indices = _.spin_exchange_indices
spin = _.spin


def loss_fn(params):
    global KEY, x_o

    def prefac(r_coords):
        rcm = jnp.mean(r_coords, axis=0)
        r = r_coords - rcm[None, :]
        delta_r1 = jnp.linalg.norm(r[0, :]) ** 2
        delta_r2 = jnp.linalg.norm(r[1, :]) ** 2
        return jnp.exp(- (delta_r1 + delta_r2) / params[0] ** 2)

    def psi(r_coords):
        return prefac(r_coords) * spin

    def weight_function(r_coords):
        psi_r = psi(r_coords)
        return jnp.vdot(psi_r, psi_r)

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


opt_init, opt_update, get_params = optimizers.adam(LEARNING_RATE)
opt_state = opt_init(jnp.array([1.32]))


def step(step, opt_state):
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    grads.block_until_ready()
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


for i in range(N_OPTIMIZATION_STEPS):
    value, opt_state = step(i, opt_state)
    print(i, value, get_params(opt_state))
