#include "kernel.cuh"

__global__ void helloKernel()
{
    printf("Hello World from GPU!\n");
}

void helloFromGPU()
{

    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}

// __global__ void sparseMatrixVecDotKernel(const float *val, const int *colInd, const int *indexPtr,
//                                          const float *vec, float *result, int startRow, int numRows, int vecSize)
// {
//     int row = blockIdx.x * blockDim.x + threadIdx.x + startRow;
//     // extern __shared__ float sharedVec[];

//     // for (int i = threadIdx.x; i < vecSize; i += blockDim.x)
//     // {
//     //     sharedVec[i] = vec[i];
//     // }
//     // __syncthreads();

//     if (row < numRows)
//     {
//         float dotProduct = 0.0f;
//         for (int j = indexPtr[row]; j < indexPtr[row + 1]; j++)
//         {
//             dotProduct += val[j] * vec[colInd[j]];
//         }
//         result[row - startRow] = dotProduct;
//     }
// }

__global__ void sparseMatrixVecDotKernel(const float *val, const int *colInd, const int *indexPtr,
                                         const float *vec, float *result, int numRows, int vecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用共享内存来存储 vec 向量（如果大小适合）
    // extern __shared__ float sharedVec[];
    // if (threadIdx.x < vecSize)
    // {
    //     sharedVec[threadIdx.x] = vec[threadIdx.x];
    // }
    // __syncthreads(); // 确保所有数据都加载到 sharedVec

    if (row < numRows)
    {
        float dotProduct = 0.0f;
        for (int j = indexPtr[row]; j < indexPtr[row + 1]; j++)
        {
            dotProduct += val[j] * vec[colInd[j]];
        }
        result[row] = dotProduct;
    }
}
void kernelSparseMatVecdot(const std::vector<float> &val,
                           const std::vector<int> &colInd,
                           const std::vector<int> &indexPtr,
                           const std::vector<float> &vec,
                           std::vector<float> &result)
{
    // Check for valid inputs (omitted for brevity)

    // Allocate memory on GPU
    float *d_val, *d_vec, *d_result;
    int *d_colInd, *d_indexPtr;
    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    // Use cudaMalloc to allocate memory (omitted for brevity)
    cudaMalloc(&d_val, val.size() * sizeof(float));
    cudaMalloc(&d_colInd, colInd.size() * sizeof(int));
    cudaMalloc(&d_indexPtr, indexPtr.size() * sizeof(int));
    cudaMalloc(&d_vec, vec.size() * sizeof(float));
    cudaMalloc(&d_result, result.size() * sizeof(float));
    // Copy data to GPU

    cudaMemcpy(d_val, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, colInd.data(), colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indexPtr, indexPtr.data(), indexPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice);
    auto sizeVec = vec.size();
    // Use cudaMemcpy (omitted for brevity)

    // Calculate grid and block sizes
    int blockSize = 256; // Example block size, adjust as needed
    int numBlocks = (indexPtr.size() + blockSize - 1) / blockSize;

    // Launch kernel
    auto start = std::chrono::steady_clock::now();
    sparseMatrixVecDotKernel<<<numBlocks, blockSize>>>(d_val, d_colInd, d_indexPtr, d_vec, d_result, indexPtr.size() - 1, vec.size());

    // Copy results back to host
    cudaMemcpy(result.data(), d_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    printf("kernelSparseMatVecdot time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
    // Clean up, free GPU memory
    cudaFree(d_val);
    cudaFree(d_colInd);
    cudaFree(d_indexPtr);
    cudaFree(d_vec);
    cudaFree(d_result);
}
// void kernelSparseMatVecdot(const std::vector<float> &val,
//                            const std::vector<int> &colInd,
//                            const std::vector<int> &indexPtr,
//                            const std::vector<float> &vec,
//                            std::vector<float> &result)
// {
//     // 省略输入检查和内存分配代码
//     float *d_val, *d_vec, *d_result;
//     int *d_colInd, *d_indexPtr;
//     // Use cudaMalloc to allocate memory (omitted for brevity)
//     cudaMalloc(&d_val, val.size() * sizeof(float));
//     cudaMalloc(&d_colInd, colInd.size() * sizeof(int));
//     cudaMalloc(&d_indexPtr, indexPtr.size() * sizeof(int));
//     cudaMalloc(&d_vec, vec.size() * sizeof(float));
//     cudaMalloc(&d_result, result.size() * sizeof(float));
//     // Copy data to GPU
//     cudaMemcpy(d_val, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_colInd, colInd.data(), colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_indexPtr, indexPtr.data(), indexPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice);
//     auto sizeVec = vec.size();
//     // 初始化 cuSPARSE
//     cusparseHandle_t handle;
//     cusparseCreate(&handle);
//     // 准备矩阵描述符
//     cusparseSpMatDescr_t matA;
//     cusparseCreateCsr(&matA, indexPtr.size() - 1, vec.size(), val.size(),
//                       d_indexPtr, d_colInd, d_val,
//                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

//     // 准备向量描述符
//     cusparseDnVecDescr_t vecX, vecY;
//     cusparseCreateDnVec(&vecX, vec.size(), d_vec, CUDA_R_32F);
//     cusparseCreateDnVec(&vecY, indexPtr.size() - 1, d_result, CUDA_R_32F);

//     // 执行矩阵向量乘法
//     float alpha = 1.0f;
//     float beta = 0.0f;
//     void *dBuffer = NULL;
//     size_t bufferSize = 0;
//     cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                             &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                             CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
//     cudaMalloc(&dBuffer, bufferSize);
//     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                  CUSPARSE_MV_ALG_DEFAULT, dBuffer);
//     cudaGetLastError();
//     cudaDeviceSynchronize();
//     // 将结果复制回主机
//     cudaMemcpy(result.data(), d_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

//     // 清理资源
//     cusparseDestroySpMat(matA);
//     cusparseDestroyDnVec(vecX);
//     cusparseDestroyDnVec(vecY);
//     cusparseDestroy(handle);
//     // 省略释放 GPU 内存的代码
//     cudaFree(d_val);
//     cudaFree(d_colInd);
//     cudaFree(d_indexPtr);
//     cudaFree(d_vec);
//     cudaFree(d_result);
// }
// void kernelSparseMatVecdot(const std::vector<float> &val,
//                            const std::vector<int> &colInd,
//                            const std::vector<int> &indexPtr,
//                            const std::vector<float> &vec,
//                            std::vector<float> &result)
// {
//     int M = 4;
//     int blockSize = 256;
//     int numRows = indexPtr.size() - 1;
//     int numBlocks = (numRows + blockSize - 1) / blockSize;

//     int rowsPerStream = numRows / M;
//     int remainingRows = numRows % M;

//     // 创建 CUDA 流
//     cudaStream_t streams[M];
//     for (int i = 0; i < M; ++i)
//     {
//         cudaStreamCreate(&streams[i]);
//     }

//     // 在设备上分配内存
//     float *d_val, *d_vec, *d_result;
//     int *d_colInd, *d_indexPtr;
//     cudaMalloc(&d_val, val.size() * sizeof(float));
//     cudaMalloc(&d_colInd, colInd.size() * sizeof(int));
//     cudaMalloc(&d_indexPtr, indexPtr.size() * sizeof(int));
//     cudaMalloc(&d_vec, vec.size() * sizeof(float));
//     cudaMalloc(&d_result, numRows * sizeof(float));
//     cudaHostRegister(const_cast<float *>(val.data()), val.size() * sizeof(float), cudaHostRegisterDefault);
//     cudaHostRegister(const_cast<int *>(colInd.data()), colInd.size() * sizeof(int), cudaHostRegisterDefault);
//     cudaHostRegister(const_cast<int *>(indexPtr.data()), indexPtr.size() * sizeof(int), cudaHostRegisterDefault);
//     cudaHostRegister(const_cast<float *>(vec.data()), vec.size() * sizeof(float), cudaHostRegisterDefault);
//     cudaHostRegister(result.data(), vec.size() * sizeof(float), cudaHostRegisterDefault);
//     auto sizeVec = vec.size();
//     // 将数据拷贝到设备
//     cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice);

//     for (int i = 0; i < M; ++i)
//     {
//         int startRow = i * rowsPerStream;
//         int endRow = (i != M - 1) ? startRow + rowsPerStream : numRows;

//         size_t valStart = indexPtr[startRow];
//         size_t valEnd = indexPtr[endRow];
//         cudaMemcpyAsync(d_val + valStart, val.data() + valStart, (valEnd - valStart) * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(d_colInd + valStart, colInd.data() + valStart, (valEnd - valStart) * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
//         cudaMemcpyAsync(d_indexPtr + startRow, indexPtr.data() + startRow, (endRow - startRow + 1) * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

//         sparseMatrixVecDotKernel<<<numBlocks / M + 1, blockSize, 0, streams[i]>>>(d_val, d_colInd, d_indexPtr, d_vec, d_result + startRow, startRow, endRow, sizeVec);
//         // sizeVec * sizeof(float)
//         cudaMemcpyAsync(result.data() + startRow, d_result + startRow, (endRow - startRow) * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
//     }

//     // 等待所有流完成
//     for (int i = 0; i < M; ++i)
//     {
//         cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     // 释放设备内存
//     cudaHostUnregister(const_cast<float *>(val.data()));
//     cudaHostUnregister(const_cast<int *>(colInd.data()));
//     cudaHostUnregister(const_cast<int *>(indexPtr.data()));
//     cudaHostUnregister(const_cast<float *>(vec.data()));
//     cudaHostUnregister(result.data());
//     cudaFree(d_val);
//     cudaFree(d_colInd);
//     cudaFree(d_indexPtr);
//     cudaFree(d_vec);
//     cudaFree(d_result);
// }
