from sympy.physics.quantum.cg import CG

from sympy import S


def get_spherical_harmonics(total_orbital_angular_momentum=0, total_z_orbital_angular_momentum=0):
    l_z_total = total_z_orbital_angular_momentum
    l_total = total_orbital_angular_momentum
    l_1 = S(1) / 2
    l_z_1 = S(1) / 2
    l_2 = S(1) / 2
    l_z_2 = S(1) / 2
    cg = CG(l_1, l_z_1,
            l_2, l_z_2,
            l_total, l_z_total
            )
    spherical_harmonics = cg.doit()
    return spherical_harmonics
