def build_wave_function(build_functions, args, kwargs):
    wave_function_list = [build_function(*arg, **kwarg) for build_function, arg, kwarg in
                          zip(build_functions, args, kwargs)]

    def wave_function(parameters, r_coords, current_wave_function=1.0, parameter_index=0):
        for wfc in wave_function_list:
            parameters, r_coords, current_wave_function, parameter_index = wfc(parameters
                                                                               , r_coords
                                                                               , current_wave_function
                                                                               , parameter_index)
            return parameters, r_coords, current_wave_function, parameter_index

    return wave_function
