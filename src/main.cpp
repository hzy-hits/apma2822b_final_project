#include <iostream>

#include "kernel.cuh"
#include "particle.h"

int main()
{

    std::cout << "Hello, World!" << std::endl;

    ParticleSystem ps(1024 * 24 * 16, false);

    ps.randomWalk();
    std::cout << "Bye, World!" << std::endl;
    return 0;
}
