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
        for (size_t flag = 0; flag < 1; flag += 1)
        {
            grid.resize(colMat * colMat, 0);
            auto start = std::chrono::high_resolution_clock::now();
            startRandomWalk();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Computing time: " << duration.count() << "ms \t" << std::endl;
            std::string filename1 = "../data/gpu/data_" + std::to_string(flag) + ".txt";
            std::ofstream out1(filename1);
            for (size_t j = 0; j < particlesPosition.size(); j += 3)
            {
                out1 << particlesPosition[j] << " " << particlesPosition[j + 1] << " " << particlesPosition[j + 2] << std::endl;

                dealwithGrid(particlesPosition[j], particlesPosition[j + 1]);
            }
            out1.close();
            std::string filename = "../data/gpu/result_" + std::to_string(flag) + ".txt";
            std::ofstream out(filename);
            for (int col = 0; col < colMat; col++)
            {
                for (int row = 0; row < colMat; row++)
                {

                    out << grid[row * colMat + col] << " ";
                }
                out << std::endl;
            }
            out.close();
            grid.clear();
        }
    }

private:
    sparseMatrix sparseMat;
    void processParticle(int particleIndex,
                         sparseMatrix &localSparseMat);
    float bilinearInterpolate(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
    {
        float x2x1 = x2 - x1;
        float y2y1 = y2 - y1;
        float x2x = x2 - x;
        float y2y = y2 - y;
        float yy1 = y - y1;
        float xx1 = x - x1;

        return q11 * x2x * y2y / (x2x1 * y2y1) +
               q21 * xx1 * y2y / (x2x1 * y2y1) +
               q12 * x2x * yy1 / (x2x1 * y2y1) +
               q22 * xx1 * yy1 / (x2x1 * y2y1);
    }
    void processBlock(int blockIndex, int numParticles);
    void startRandomWalk();
    void init_particles();
    void classic_random_walk(float &new_x, float &new_y);

    std::vector<float> sparseMatrixVecDot(const sparseMatrix &mat,
                                          const std::vector<float> &vec);
    std::vector<float> sparseMatrixVecDotGpu(const sparseMatrix &mat,
                                             const std::vector<float> &vec);

    void parallelProcessParticles();
    void dealwithGrid(float particleX, float particleY)
    {
        // 根据粒子的原始坐标，将其移动到矩阵中心
        particleX += 801;
        particleY += 801;

        // 找到粒子周围的四个格点
        int x1 = static_cast<int>(particleX);
        int y1 = static_cast<int>(particleY);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        // 检查粒子是否在 grid 的边界附近
        if (x1 < 0 || y1 < 0 || x2 >= colMat || y2 >= colMat)
        {
            std::cout << "Particle coordinates are out of grid bounds" << std::endl;
            return;
        }

        // 计算每个格点的权重
        float weight11 = (x2 - particleX) * (y2 - particleY);
        float weight21 = (particleX - x1) * (y2 - particleY);
        float weight12 = (x2 - particleX) * (particleY - y1);
        float weight22 = (particleX - x1) * (particleY - y1);

        // 更新每个格点的值，同时考虑边界情况
        if (x1 >= 0 && y1 >= 0 && x1 < colMat && y1 < colMat)
            grid[x1 * colMat + y1] += weight11;
        if (x2 >= 0 && y1 >= 0 && x2 < colMat && y1 < colMat)
            grid[x2 * colMat + y1] += weight21;
        if (x1 >= 0 && y2 >= 0 && x1 < colMat && y2 < colMat)
            grid[x1 * colMat + y2] += weight12;
        if (x2 >= 0 && y2 >= 0 && x2 < colMat && y2 < colMat)
            grid[x2 * colMat + y2] += weight22;
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
    int colMat = 1601;
    std::vector<float> grid;

    // std::random_device rd;
    // std::mt19937 m_gen;
    // std::uniform_real_distribution<float> piDistribution(0.f, 2.f * M_PI);
};
