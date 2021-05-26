import jax.numpy as jnp
from operator import mul, add, sub

operator_str = {mul: '*', add: '+', sub: '-'}


def combine_wave_functions(func_1, params_1, name_1, func_2, params_2, name_2, operator):
    split_index = len(params_1)

    def combined_psi_prefactor(combined_params, r_coords):
        f_1_r = func_1(combined_params[:split_index], r_coords)
        f_2_r = func_2(combined_params[split_index:], r_coords)
        return operator(f_1_r, f_2_r)

    combined_parameters = jnp.concatenate((params_1, params_2))
    return combined_psi_prefactor, combined_parameters, name_1 + operator_str[operator] + name_2
