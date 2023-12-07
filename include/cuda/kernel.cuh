#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdio.h>

#include <vector>

void helloFromGPU();
void kernelSparseMatVecdot(const std::vector<float> &val,
                           const std::vector<int> &colInd,
                           const std::vector<int> &indexPtr,
                           const std::vector<float> &vec,
                           std::vector<float> &result);
#endif
