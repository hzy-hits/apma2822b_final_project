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
#include <unordered_map>
#include <utility>
using GridCell = std::pair<int, int>;
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
    ParticleSystem(int particles_number, bool isCuda) : num_particles(particles_number), flag(isCuda) { init_particles(); };
    void randomWalk()
    {
        for (size_t i = 0; i < 2; i += 1)
        {

            auto start = std::chrono::high_resolution_clock::now();
            startRandomWalk();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Computing time: " << duration.count() << "ms \t" << std::endl;
#pragma omp parallel for
            for (size_t j = 0; j < particlesPosition.size(); j += 3)
            {
                dealwithGrid(particlesPosition[j], particlesPosition[j + 1]);
            }

            std::string filename = "../data/gpu/result_" + std::to_string(i) + ".txt";
            std::ofstream out(filename);
            for (size_t i = 0; i < colMat; i++)
            {
                for (size_t j = 0; j < colMat; j++)
                {
                    out << grid[i * colMat + j] << " ";
                }
                out << std::endl;
            }
            out.close();
            reSetGrid();
        }
    }

private:
    sparseMatrix sparseMat;
    void processParticle(int particleIndex,
                         sparseMatrix &localSparseMat);
    void processBlock(int blockIndex, int numParticles);
    void startRandomWalk();
    void init_particles();
    void classic_random_walk(float &new_x, float &new_y);

    std::vector<float> sparseMatrixVecDot(const sparseMatrix &mat,
                                          const std::vector<float> &vec);
    std::vector<float> sparseMatrixVecDotGpu(const sparseMatrix &mat,
                                             const std::vector<float> &vec);

    void parallelProcessParticles();
    void dealwithGrid(float x, float y)
    {
        int newX = round(2 * x + 800);
        int newY = round(2 * y + 800);
        grid[newX * colMat + newY] += 1;
    }
    void reSetGrid()
    {
        for (int i = 0; i < grid.size(); i++)
        {
            grid[i] = 0;
        }
    }
    int num_particles;
    int numBlocks;
    std::vector<sparseMatrix> sparseMatsBlocks;
    std::vector<std::vector<float>> particlesPositionBlocks;
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
    bool flag = false;

    std::vector<int> grid{2563201, 0};
    int colMat = 1601;
    // std::random_device rd;
    // std::mt19937 m_gen;
    // std::uniform_real_distribution<float> piDistribution(0.f, 2.f * M_PI);
};
