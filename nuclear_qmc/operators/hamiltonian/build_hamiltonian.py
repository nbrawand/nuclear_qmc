from nuclear_qmc.operators.hamiltonian.arxiv_2007_14282v2.potential_energy import build_arxiv_2007_14282v2
from nuclear_qmc.operators.hamiltonian.arxiv_2102_02327v1.potential_energy import build_arxiv_2102_02327v1
from nuclear_qmc.operators.operators import kinetic_energy_psi


def build_hamiltonian(potential_energy_expression, particle_pairs, particle_triplets, spin_exchange_indices,
                      isospin_exchange_indices, potential_kwargs=None):
    if potential_kwargs is None:
        potential_kwargs = {}

    potentials = {
        'arxiv_2007_14282v2': {
            'builder': build_arxiv_2007_14282v2,
            'args': [particle_pairs, particle_triplets,
                     spin_exchange_indices],
        },
        'arxiv_2102_02327v1': {
            'builder': build_arxiv_2102_02327v1,
            'args': [particle_pairs, spin_exchange_indices, isospin_exchange_indices]
        }
    }

    builder = potentials[potential_energy_expression]['builder']
    args = potentials[potential_energy_expression]['args']
    potential_energy = builder(*args, **potential_kwargs)

    def hamiltonian(psi, psi_params, psi_vector, r_coords):
        ke_psi = kinetic_energy_psi(psi, psi_params, r_coords) * psi_vector
        v_psi = potential_energy(psi, psi_params, psi_vector, r_coords)
        h_psi = ke_psi + v_psi
        return h_psi

    return hamiltonian
