import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from typing import NoReturn


def verify_gpu_computation() -> NoReturn:
    # --- FIX 1: Initialize the driver ---
    cuda.init()

    # Get device properties
    device = cuda.Device(0)

    # --- FIX 2: Create and push a context ---
    # Memory allocation requires an active context on the device
    ctx = device.make_context()

    try:
        device_name = device.name()
        total_memory = device.total_memory()

        print(f"Successfully initialized CUDA on: {device_name}")
        print(f"Total Memory: {total_memory / 1024 ** 2:.2f} MB")

        # Define the size of the arrays
        vector_length = 400

        # Create host arrays
        host_a = np.random.randn(vector_length).astype(np.float32)
        host_b = np.random.randn(vector_length).astype(np.float32)

        # Allocate memory on the device
        device_a = cuda.mem_alloc(host_a.nbytes)
        device_b = cuda.mem_alloc(host_b.nbytes)
        device_c = cuda.mem_alloc(host_a.nbytes)

        # Copy data from host to device
        cuda.memcpy_htod(device_a, host_a)
        cuda.memcpy_htod(device_b, host_b)

        # Define the CUDA kernel
        cuda_kernel_code = """
        __global__ void add_vectors(float *a, float *b, float *c, int n)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n)
            {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        # Compile the kernel
        module = SourceModule(cuda_kernel_code)
        add_vectors_func = module.get_function("add_vectors")

        # Launch the kernel
        threads_per_block = 256
        blocks_per_grid = (vector_length + threads_per_block - 1) // threads_per_block

        add_vectors_func(
            device_a, device_b, device_c,
            np.int32(vector_length),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1)
        )

        # Get results
        host_c = np.empty_like(host_a)
        cuda.memcpy_dtoh(host_c, device_c)

        # Verify
        expected_result = host_a + host_b
        if np.allclose(host_c, expected_result):
            print("GPU computation successful! The results match.")
        else:
            print("Error: Results mismatch.")

    finally:
        # --- FIX 3: Clean up the context ---
        # It is good practice to pop the context when done
        ctx.pop()


if __name__ == '__main__':
    try:
        verify_gpu_computation()
    except cuda.Error as e:
        print(f"An error occurred with CUDA: {e}")