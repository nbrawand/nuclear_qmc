# Parallelization
Current parallelization is done at the block level. If the input value "number_of_parallel_devices" is set to a value
larger than zero, the blocks are computed sequentially in step sizes equal to the number of parallel devices. Each block
within the step is pmapped across the available devices.

Memory requirements can be reduced by reducing the number of walkers per block while increasing the number of blocks
during sampling. This trades space complexity for time complexity. Add additional GPUs will allow more blocks
to be computed in parallel and reduce execution time.

If "number_of_parallel_devices" is zero (default value) all blocks are vmapped. Setting the number of devices to 1
will compute each block within a for loop in steps of 1 block at a time on a single gpu.
