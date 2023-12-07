#include <iostream>
#include <chrono>
#include "kernel.cuh"
#include "particle.h"

int main()
{

    std::cout << "Hello, World!" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ParticleSystem ps(1024 * 24 * 10);
    ps.randomWalk();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << duration.count() << "ms \t"
              << "Bye, World!" << std::endl;
    return 0;
}
