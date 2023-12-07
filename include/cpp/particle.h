#pragma once
#include <vector>
#include <string>
#include <random>
#include <omp.h>
#include <cmath>
#include "kernel.cuh"
#include <iostream>
#include <fstream>
#include <chrono>
struct sparseMatrix
{
    std::vector<float> val;
    std::vector<int> indexPtr;
    std::vector<int> colInd;

    // 合并另一个稀疏矩阵到当前矩阵
    void merge(const sparseMatrix &other)
    {
        if (!other.val.empty())
        {
            // 追加值和列索引
            val.insert(val.end(), other.val.begin(), other.val.end());
            colInd.insert(colInd.end(), other.colInd.begin(), other.colInd.end());

            // 追加行指针，需对除第一个元素外的每个元素加上当前val的大小
            for (size_t i = 1; i < other.indexPtr.size(); ++i)
            {
                indexPtr.push_back(other.indexPtr[i] + val.size());
            }
        }
    }
    void reserve(int size, int row)
    {
        val.reserve(row * size);
        indexPtr.reserve(row * size);
        colInd.reserve(size + 1);
    }
    // 清空稀疏矩阵
    void clear()
    {
        val.clear();
        indexPtr.clear();
        colInd.clear();
    }
};

struct particle2D
{
    std::vector<float> position =
        std::vector<float>{0, 0, 1};
    // std::vector<float> velocity[2];
    std::vector<float> matrix = std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1};
};
class ParticleSystem
{
public:
    ParticleSystem(int particles_number,bool isCuda) : num_particles(particles_number),flag(isCuda) { init_particles(); };
    void randomWalk()
    {
        for (size_t i = 0; i < 1; i += 1)
        {
            std::string filename = "../data/gpu/result_" + std::to_string(i) + ".txt";
            std::ofstream out(filename);
            
            startRandomWalk();
            
            for (size_t j = 0; j < particlesPosition.size(); j += 3)
            {
                out << particlesPosition[j] << "," << particlesPosition[j + 1]
                    << std::endl;
            }
            out.close();
        }
    }

private:
    sparseMatrix sparseMat;

    void processParticle(int particleIndex,
                         sparseMatrix &localSparseMat);
    void startRandomWalk();
    void init_particles();
    void classic_random_walk(float &new_x, float &new_y);
    void sparseMatrixVecDot(const sparseMatrix &mat,
                            const std::vector<float> &vec,
                            std::vector<float> &result);
    std::vector<float> sparseMatrixVecDotGpu(const sparseMatrix &mat,
                                             const std::vector<float> &vec);

    void parallelProcessParticles();

    int num_particles;
    // std::vector<sparseMatrix> localSparseMats;
    // std::vector<particle2D> particles;
    // std::vector<std::vector<float>> particlesMatrix;
    std::vector<float>
        particlesPosition;
    int maxThreads;
    int step;
    int particlePerThreads;
    int remainParticle;
    int maxParticlesPerBlock;
    std::vector<float> Posresult;
    bool flag=false;
    float time;
    // std::random_device rd;
    // std::mt19937 m_gen;
    // std::uniform_real_distribution<float> piDistribution(0.f, 2.f * M_PI);
};
