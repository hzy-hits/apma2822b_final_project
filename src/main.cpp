#include <iostream>
#include <thread>
#include "kernel.cuh"
#include "particle.h"
#include <chrono>
int main()
{

    std::cout << "Hello, World!" << std::endl;
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Number of threads: " << n << std::endl;
    ParticleSystem ps(1024 * 32, true);
    ps.randomWalk();

    std::cout << "Bye, World!" << std::endl;
    return 0;
}
