from sympy.physics.quantum.cg import CG
import jax.numpy as jnp
from scipy.special import sph_harm
import numpy as np
from sympy import S

from nuclear_qmc.wave_function.neural_network_jastro_builder.build_nn_wave_function import build_nn_wave_function


def sigmoid(x):
    return 1. / (1 + jnp.exp(-x))


def build_radial_function(key
                          , n_dense
                          , n_hidden_layers
                          , nn_wrapper_function=sigmoid
                          ):
    key, nn, params = build_nn_wave_function(ndense=n_dense, key=key, n_hidden_layers=n_hidden_layers)

    def func(p, r_i):
        r_mag = jnp.linalg.norm(r_i)
        out = nn(p, r_mag)
        out = nn_wrapper_function(out)
        return out

    return key, func, params


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


def Y1m1(r):
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    return sq34pi * r[1]


def Y11(r):
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    return sq34pi * r[0]


def Y10(r):
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    return sq34pi * r[2]


def Y00(r):
    return 1.0


SPHERICAL_HARMONICS = {'Y11': Y11, 'Y1m1': Y1m1, 'Y10': Y10, 'Y00': Y00}


def get_spherical_harmonic_function(L, L_z):
    sq12pi = jnp.sqrt(1. / jnp.pi) / 2.
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    if L_z == 0 and L == 0:
        def func(r_i):
            return sq12pi
    elif L == 1 and L_z == -1:
        def func(r_i):
            return sq34pi * r_i[1] / jnp.linalg.norm(r_i)
    elif L == 1 and L_z == 0:
        def func(r_i):
            return sq34pi * r_i[2] / jnp.linalg.norm(r_i)
    elif L == 1 and L_z == 1:
        def func(r_i):
            return sq34pi * r_i[0] / jnp.linalg.norm(r_i)
    else:
        raise RuntimeError(f'get_spherical_harmonic_function: invalid L and Lz: {L}, {L_z}')
    return func


def get_spherical_harmonic_function(L, L_z):
    sq12pi = jnp.sqrt(1. / jnp.pi) / 2.
    sq34pi = jnp.sqrt(3. / (4. * jnp.pi))
    if L_z == 0 and L == 0:
        def func(r_i):
            return sq12pi
    elif L == 1 and L_z == -1:
        def func(r_i):
            return sq34pi * r_i[1] / jnp.linalg.norm(r_i)
    elif L == 1 and L_z == 0:
        def func(r_i):
            return sq34pi * r_i[2] / jnp.linalg.norm(r_i)
    elif L == 1 and L_z == 1:
        def func(r_i):
            return sq34pi * r_i[0] / jnp.linalg.norm(r_i)
    else:
        raise RuntimeError(f'get_spherical_harmonic_function: invalid L and Lz: {L}, {L_z}')
    return func


def get_spherical_harmonic_functions(key, names, radial_func):
    L = [int(n.split('_')[1]) for n in names]
    Lz = [int(n.split('_')[-1]) for n in names]
    functions = [get_spherical_harmonic_function(l, lz) for n, l, lz in zip(names, L, Lz)]
    if 1 in L:
        functions = [lambda p, r: radial_func(p, r) * f(r) if l == 1 else lambda p, r,: f(r) for f, l in
                     zip(functions, L)]
    else:
        functions = [lambda p, r: f(r) for f, l in zip(functions, L)]

    return key, functions


def get_spherical_harmonic_system(key, L_total, L_z_total, L_1, L_2, radial_func):
    L_z_combinations = get_valid_L_z_combintations(L_z_total, L_1, L_2)
    spherical_harmonics_names = get_spherical_harmonic_names(L_1, L_2, L_z_combinations)
    coefficients = get_clebsch_gordan_coeff(L_total, L_z_total, L_1, L_2, L_z_combinations)
    functions = []
    for func_names in spherical_harmonics_names:
        key, funcs = get_spherical_harmonic_functions(key, func_names, radial_func)
        functions.append(funcs)
    return key, coefficients, functions


def get_spherical_harmonic_systems(key, n_particles, L_total, L_z_total, L_1, L_2, n_dense, n_hidden_layers):
    if n_particles <= 4:
        coefficients = jnp.array([[1.0]], dtype=jnp.float64)
        functions = n_particles * ['Y_0_0']
        key, functions = get_spherical_harmonic_functions(key, functions, None)
        functions = [functions]
        params = jnp.array([])
    elif n_particles <= 8:
        # add radial functions
        key, radial_func, params = build_radial_function(key
                                                         , n_dense
                                                         , n_hidden_layers
                                                         , nn_wrapper_function=jnp.exp)

        key, alpha_functions = get_spherical_harmonic_functions(key, 4 * ['Y_0_0'], None)
        key, coefficients, functions = get_spherical_harmonic_system(key, L_total, L_z_total, L_1, L_2, radial_func)
        functions = [f + alpha_functions for f in functions]  # add alpha core functions
    else:
        raise RuntimeError('get_spherical_harmonic_systems: n_particles must be <= 8.')

    return key, coefficients, functions, params
