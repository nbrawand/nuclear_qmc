from sympy.physics.quantum.cg import CG
import jax.numpy as jnp
from scipy.special import sph_harm
import numpy as np
from sympy import S


def get_possible_L_z_values(L):
    L_z = np.arange(-L, 2 * L, L)
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


def get_spherical_harmonic_function(L, L_z):
    return lambda theta, phi: sph_harm(L_z, L, theta, phi)


def get_spherical_harmonic_functions(L_1, L_2, L_z_combinations):
    functions = [[get_spherical_harmonic_function(L_1, lz1), get_spherical_harmonic_function(L_2, lz2)]
                 for lz1, lz2 in L_z_combinations]
    return np.array(functions)


def get_spherical_harmonic_system(L_total, L_z_total, L_1, L_2):
    L_z_combinations = get_valid_L_z_combintations(L_z_total, L_1, L_2)
    spherical_harmonics_names = get_spherical_harmonic_names(L_1, L_2, L_z_combinations)
    coefficients = get_clebsch_gordan_coeff(L_total, L_z_total, L_1, L_2, L_z_combinations)
    functions = get_spherical_harmonic_functions(L_1, L_2, L_z_combinations)
    return spherical_harmonics_names, coefficients, functions
