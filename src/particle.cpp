#include "particle.h"

void ParticleSystem::init_particles()

{

    maxParticlesPerBlock = 1024 * 4;
    numBlocks = num_particles / maxParticlesPerBlock;
    remainParticle = num_particles % maxParticlesPerBlock;

    Posresult.reserve(3 * num_particles);
    particlesPosition.reserve(3 * num_particles);

    // sparseMatsBlocks.reserve(counter);
    // particlesPositionBlocks.reserve(counter);
    step = 400;
    maxThreads = omp_get_max_threads();
    // localSparseMats.resize(omp_get_max_threads());
    // remainParticle = num_particles % maxParticlesPerBlock;
    for (int i = 0; i < num_particles; i++)
    {
        particlesPosition.push_back(0);
        particlesPosition.push_back(0);
        particlesPosition.push_back(1);
    }
}
void ParticleSystem::parallelProcessParticles()
{
    for (int i = 0; i < numBlocks; i++)
    {
        processBlock(i, maxParticlesPerBlock);
    }
    if (remainParticle > 0)
    {
        processBlock(numBlocks, remainParticle);
    }
}
void ParticleSystem::processBlock(int blockIndex, int numParticles)
{

    int startIdx = blockIndex * maxParticlesPerBlock * 3;
    int endIdx = std::min(startIdx + numParticles * 3, static_cast<int>(particlesPosition.size()));

    std::vector<float> localParticlesPosition(particlesPosition.begin() + startIdx, particlesPosition.begin() + endIdx);
    particlesPositionBlocks.push_back(localParticlesPosition);

    sparseMatrix localSparseMat;
    localSparseMat.indexPtr.push_back(0);

    for (int j = 0; j < numParticles; j++)
    {
        processParticle(j, localSparseMat);
    }

    sparseMatsBlocks.push_back(localSparseMat);
}
void ParticleSystem::startRandomWalk()
{
    parallelProcessParticles();

    if (flag)
    {
        int counter = remainParticle == 0 ? numBlocks : numBlocks + 1;
#pragma omp parallel
        {
            std::vector<float> localPosresult;
#pragma omp for nowait
            for (int i = 0; i < counter; i++)
            {
                auto &element1 = sparseMatsBlocks[i];
                auto &element2 = particlesPositionBlocks[i];
                std::vector<float> localNewPos = sparseMatrixVecDotGpu(element1, element2);
                localPosresult.insert(localPosresult.end(), localNewPos.begin(), localNewPos.end());
            }
#pragma omp critical
            Posresult.insert(Posresult.end(), localPosresult.begin(), localPosresult.end());
        }
        particlesPosition = Posresult;
        Posresult.clear();
        sparseMatsBlocks.clear();
        particlesPositionBlocks.clear();
    }
    else
    {
        int counter = sparseMatsBlocks.size();

#pragma omp parallel
        {
            std::vector<float> localPosresult;
#pragma omp for nowait
            for (int i = 0; i < counter; i++)
            {
                const auto &element1 = sparseMatsBlocks[i];
                const auto &element2 = particlesPositionBlocks[i];
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<float> localNewPos = sparseMatrixVecDot(element1, element2);
                localPosresult.insert(localPosresult.end(), localNewPos.begin(), localNewPos.end());
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                std::cout << "kernel time: " << duration.count() << "ms \t" << std::endl;
            }
#pragma omp critical
            Posresult.insert(Posresult.end(), localPosresult.begin(), localPosresult.end());
        }
        particlesPosition = Posresult;
        Posresult.clear();
        sparseMatsBlocks.clear();
        particlesPositionBlocks.clear();
    }
    for (int i = 0; i < 9; i += 3)
    {
        std::cout << "result sample: " << std::endl
                  << particlesPosition[i] << "\t" << particlesPosition[i + 1]
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
std::vector<float> ParticleSystem::sparseMatrixVecDot(const sparseMatrix &mat,
                                                      const std::vector<float> &vec)
{
    std::vector<float> result;
    result.resize(mat.indexPtr.size() - 1, 0.0f);
#pragma omp parallel for
    for (size_t i = 0; i < mat.indexPtr.size() - 1; i++)
    {
        for (int j = mat.indexPtr[i]; j < mat.indexPtr[i + 1]; j++)
        {
            result[i] += mat.val[j] * vec[mat.colInd[j]];
        }
    }
    // result.push_back(1);
    return result;
}
std::vector<float> ParticleSystem::sparseMatrixVecDotGpu(const sparseMatrix &mat,
                                                         const std::vector<float> &vec)
{
    std::vector<float> result;
    result.resize(vec.size());
    std::vector<float> val = mat.val;
    std::vector<int> colInd = mat.colInd;
    std::vector<int> indexPtr = mat.indexPtr;
    // std::cout << "indexPtr.size():" << indexPtr.size() << std::endl;
    // std::cout << "val.size():" << val.size() << std::endl;
    // std::cout << "colInd.size():" << colInd.size() << std::endl;
    // std::cout << "vec.size():" << vec.size() << std::endl;

    kernelSparseMatVecdot(val, colInd, indexPtr, vec, result);

    return result;
}
