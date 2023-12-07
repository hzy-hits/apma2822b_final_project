#include "particle.h"

void ParticleSystem::init_particles()

{

    // maxParticlesPerBlock = 2048;
    Posresult.reserve(3 * num_particles);
    step = 400;
    maxThreads = omp_get_max_threads();
    // localSparseMats.resize(omp_get_max_threads());
    // remainParticle = num_particles % maxParticlesPerBlock;
    particlesPosition.reserve(3 * num_particles);
    for (int i = 0; i < num_particles; i++)
    {
        particlesPosition.push_back(0);
        particlesPosition.push_back(0);
        particlesPosition.push_back(1);
    }
}

void ParticleSystem::parallelProcessParticles()
{
    sparseMat.clear();
    std::vector<sparseMatrix> localSparseMats(omp_get_max_threads());

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto &localSparseMat = localSparseMats[thread_id];
        localSparseMat.indexPtr.push_back(0);

#pragma omp for nowait
        for (int i = 0; i < num_particles; i++)
        {
            processParticle(i, localSparseMat);
        }
    }
    for (auto &localMat : localSparseMats)
    {
        if (sparseMat.indexPtr.size() == 0)
        {
            sparseMat = localMat;
        }

        sparseMat.merge(localMat);
    }
}

void ParticleSystem::startRandomWalk()
{
    parallelProcessParticles();
    // sparseMatrixVecDot(sparseMat, particlesPosition, Posresult);
    particlesPosition = sparseMatrixVecDotGpu(sparseMat, particlesPosition);
    // particlesPosition = Posresult;
    for (int i = 0; i < 9; i += 3)
    {
        std::cout << particlesPosition[i] << "\t" << particlesPosition[i + 1]
                  << "\t" << particlesPosition[i + 2]
                  << std::endl;
        // std::cout << Posresult[i] << "\t" << Posresult[i + 1]
        //           << "\t" << Posresult[i + 2]
        //           << std::endl;
    }
}

void ParticleSystem::processParticle(int particleIndex, sparseMatrix &localSparseMat)
{
    int colOffset = 3 * particleIndex;
    float new_x, new_y;
    classic_random_walk(new_x, new_y);
    localSparseMat.colInd.push_back(0 + colOffset);
    localSparseMat.val.push_back(1);
    localSparseMat.colInd.push_back(2 + colOffset);
    localSparseMat.val.push_back(new_x);
    localSparseMat.indexPtr.push_back(localSparseMat.val.size());

    localSparseMat.colInd.push_back(1 + colOffset);
    localSparseMat.val.push_back(1);
    localSparseMat.colInd.push_back(2 + colOffset);
    localSparseMat.val.push_back(new_y);
    localSparseMat.indexPtr.push_back(localSparseMat.val.size());

    localSparseMat.colInd.push_back(2 + colOffset);
    localSparseMat.val.push_back(1);
    localSparseMat.indexPtr.push_back(localSparseMat.val.size());
}

void ParticleSystem::classic_random_walk(float &new_x, float &new_y)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0, 2 * M_PI);
    float random = distribution(gen);
    new_x = cos(random);
    new_y = sin(random);
}
void ParticleSystem::sparseMatrixVecDot(const sparseMatrix &mat,
                                        const std::vector<float> &vec,
                                        std::vector<float> &result)
{
    result.clear();
    result.resize(mat.indexPtr.size() - 1, 0.0f);
#pragma omp parallel for
    for (size_t i = 0; i < mat.indexPtr.size() - 1; i++)
    {
        for (int j = mat.indexPtr[i]; j < mat.indexPtr[i + 1]; j++)
        {
            result[i] += mat.val[j] * vec[mat.colInd[j]];
        }
    }
}
std::vector<float> ParticleSystem::sparseMatrixVecDotGpu(const sparseMatrix &mat,
                                                         const std::vector<float> &vec)
{
    std::vector<float> result;
    result.resize(vec.size());
    std::vector<float> val = mat.val;
    std::vector<int> colInd = mat.colInd;
    std::vector<int> indexPtr = mat.indexPtr;
    kernelSparseMatVecdot(val, colInd, indexPtr, vec, result);
    return result;
}
