from sympy.physics.quantum.cg import CG
import jax.numpy as jnp
from scipy.special import sph_harm
import numpy as np
from sympy import S


def get_possible_L_z_values(L):
    if L > 0:
        L_z = np.arange(-L, 2 * L, L)
    else:
        L_z = [0]
    L_z = [S(l_z) for l_z in L_z]
    return L_z


def get_valid_L_z_combintations(L_z_total, L_1, L_2):
    L_z_1 = get_possible_L_z_values(L_1)
    L_z_2 = get_possible_L_z_values(L_2)
    L_z_combinations = [[l_z_1, l_z_2] for l_z_1 in L_z_1 for l_z_2 in L_z_2 if l_z_1 + l_z_2 == L_z_total]
    return L_z_combinations


def get_clebsch_gordan_coeff(L_total, L_z_total, L_1, L_2, L_z_combinations):
    coefficients = [CG(L_1, lz1,
                       L_2, lz2,
                       L_total, L_z_total).doit().evalf() for lz1, lz2 in L_z_combinations]
    return jnp.array(coefficients, dtype=jnp.float64)


def get_spherical_harmonic_names(L_1, L_2, L_z_combinations):
    spherical_harmonics = [[f"Y_{L_1}_{lz1}", f"Y_{L_2}_{lz2}"] for lz1, lz2 in L_z_combinations]
    return np.array(spherical_harmonics)


def get_theta(r):
    """arctan \\frac{\\sqrt{x^2+y^2}}{z}"""
    theta = jnp.linalg.norm(r[:-1])
    theta = jnp.arctan2(theta, r[-1])
    return theta


def get_phi(r):
    """arctan(y/x)"""
    phi = jnp.arctan2(r[1], r[0])
    return phi


def get_spherical_harmonic_function(L, L_z):
    def func(r):
        theta = get_theta(r)
        phi = get_phi(r)
        out = sph_harm(L_z, L, theta, phi)
        return out

    return func


def get_spherical_harmonic_functions(names):
    names = list(set(names))
    L = [int(n.split('_')[1]) for n in names]
    Lz = [int(n.split('_')[-1]) for n in names]
    functions = [get_spherical_harmonic_function(l, lz) for n, l, lz in zip(names, L, Lz)]
    return functions


def get_spherical_harmonic_system(L_total, L_z_total, L_1, L_2):
    L_z_combinations = get_valid_L_z_combintations(L_z_total, L_1, L_2)
    spherical_harmonics_names = get_spherical_harmonic_names(L_1, L_2, L_z_combinations)
    coefficients = get_clebsch_gordan_coeff(L_total, L_z_total, L_1, L_2, L_z_combinations)
    functions = get_spherical_harmonic_functions(spherical_harmonics_names.reshape(-1))
    return spherical_harmonics_names, coefficients, functions


def get_spherical_harmonic_systems(n_particles, L_total, L_z_total, L_1, L_2):
    if n_particles <= 4:
        spherical_harmonics_names, coefficients, functions = get_spherical_harmonic_system(L_total, L_z_total, L_1, L_2)
        spherical_harmonics_names = [[[harm[0]], [harm[1]]] for harm in spherical_harmonics_names]
    elif n_particles <= 8:
        alpha_core = get_spherical_harmonic_system(0, 0, 0, 0)
        spherical_harmonics_names, coefficients, functions = get_spherical_harmonic_system(L_total, L_z_total, L_1, L_2)
        Y00 = 'Y_0_0'
        spherical_harmonics_names = [[[Y00, harm[0]], [Y00, harm[1]]] for harm in spherical_harmonics_names]
        functions.update(alpha_core[-1])
    else:
        raise RuntimeError('get_spherical_harmonic_systems: n_particles must be <= 8.')

    return spherical_harmonics_names, coefficients, functions
