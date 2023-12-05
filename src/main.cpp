#include <iostream>
#include "kernel.cuh"

int main()
{
    std::cout << "Hello, World!" << std::endl;
    helloFromGPU();
    std::cout << "Bye, World!" << std::endl;
    return 0;
}
