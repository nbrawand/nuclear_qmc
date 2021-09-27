from nuclear_qmc.operators.hamiltonian.arxiv_2007_14282v2.potential_energy import build_arxiv_2007_14282v2
from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import Arxiv_2102_02327v1_Potential
from nuclear_qmc.operators.operators import kinetic_energy_psi
from nuclear_qmc.utils.finite_difference import kinetic_energy
import jax.numpy as jnp


def build_hamiltonian(potential_energy_expression, particle_pairs, particle_triplets, spin_exchange_indices,
                      isospin_exchange_indices, isospin_binary_representation=None, potential_kwargs=None
                      , use_finite_diff=True):
    """Main routine for building the hamiltonian function used for returning H|psi>"""
    if potential_kwargs is None:
        potential_kwargs = {}

    potentials = {
        'arxiv_2007_14282v2': {
            'builder': build_arxiv_2007_14282v2,
            'args': [particle_pairs, particle_triplets,
                     spin_exchange_indices],
        },
        'arxiv_2102_02327v1': {
            'builder': Arxiv_2102_02327v1_Potential,
            'args': [particle_pairs, particle_triplets
                , spin_exchange_indices, isospin_exchange_indices, isospin_binary_representation]
        }
    }

    builder = potentials[potential_energy_expression]['builder']
    args = potentials[potential_energy_expression]['args']
    potential_energy = builder(*args, **potential_kwargs)

    def hamiltonian(psi, psi_params, r_coords):
        if use_finite_diff:
            i = jnp.arange(r_coords.shape[0])
            j = jnp.arange(r_coords.shape[1])
            my_psi = lambda r: psi(psi_params, r)
            ke_psi = kinetic_energy(my_psi, r_coords, i, j)
        else:
            ke_psi = kinetic_energy_psi(psi, psi_params, r_coords)

        v_psi = potential_energy(psi, psi_params, r_coords)
        h_psi = ke_psi + v_psi
        return h_psi

    return hamiltonian
