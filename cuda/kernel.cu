#include "kernel.cuh"

__global__ void helloKernel()
{
    printf("Hello World from GPU!\n");
}

void helloFromGPU()
{

    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}
