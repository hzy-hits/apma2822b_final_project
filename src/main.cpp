#include <iostream>

#include "kernel.cuh"
#include "particle.h"

int main()
{

    std::cout << "Hello, World!" << std::endl;

    ParticleSystem ps(1024 * 24, true);

    ps.randomWalk();
    std::cout << "Bye, World!" << std::endl;
    return 0;
}
