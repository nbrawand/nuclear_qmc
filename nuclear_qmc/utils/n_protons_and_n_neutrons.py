import numpy as np


def get_n_protons_and_n_neutrons(orbitals):
    """calculate # of protons and neutrons from orbital input"""
    isospin_stgs = [[orb.split('_')[-1] for orb in det] for det in orbitals]
    isospin_ints = []
    # check each isospin is valid and assign int value
    for lst in isospin_stgs:
        tmp = []
        for st in lst:
            if st.lower() == 'p':
                val = 1
            elif st.lower() == 'n':
                val = 0
            else:
                raise RuntimeError(f'Isospin value {st} is not valid. All isospin values in orbitals must be n or p.')
            tmp.append(val)
        isospin_ints.append(tmp)
    isospin_ints = np.array(isospin_ints)

    # check that each determinant has the same # of particles and protons
    n_particles = len(isospin_ints[0])
    for det in isospin_ints:
        if len(det) != n_particles:
            raise RuntimeError('Number of particles must be same for all determinants.')
    n_protons = int(isospin_ints[0].sum())
    for det in isospin_ints:
        if int(det.sum()) != n_protons:
            raise RuntimeError('Number of particles must be same for all determinants.')
    n_neutrons = int(n_particles - n_protons)

    return n_protons, n_neutrons
