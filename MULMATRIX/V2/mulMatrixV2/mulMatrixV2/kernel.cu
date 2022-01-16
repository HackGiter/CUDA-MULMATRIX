
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Header.cuh"

#include <stdio.h>

__host__ void MulMat(const MATRIX, const MATRIX, MATRIX*);

int main()
{
    int n = 4;
    int m = 4;
    int width = 32 * n;
    int height = 32 * m;
    MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)), * m2 = (MATRIX*)malloc(sizeof(MATRIX)), * m3 = (MATRIX*)malloc(sizeof(MATRIX));

    InitMat(m1, width + 2, height + 4);
    InitMat(m2, height + 4, width + 2);
    InitMat(m3, height + 4, height + 4);

    SetMat(m1, 1.0);
    SetMat(m2, 1.5);

    dim3 threadsPerBlock(2, THREADSPERBLOCK);
    warmup << <1, 1 >> > ();
    MulMat(*m1, *m2, m3);
    PrintMat(*m3);

    return 0;
}

__host__ void MulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
    cudaEvent_t start, stop;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX(), device_m4 = MATRIX(), cpym1 = MATRIX(), cpym2 = MATRIX();

    int K = CEIL_DIV(m1.width, SUBBLOCKSIZE), int blockDimW = CEIL_DIV(m2.width, SUBBLOCKSIZE), blockDimH = CEIL_DIV(m1.height, SUBBLOCKSIZE);
    dim3 m1Dim(K, blockDimH);
    dim3 m2Dim(blockDimW, K);
    dim3 gridDim(blockDimW, blockDimH);
    dim3 blockDim(THREADSPERBLOCK / 32, THREADSPERBLOCK / 8);
 
    cpym1.width = m1.width; cpym1.height = m1.height; cpym1.size = m1.size;
    CHECK_CUDA(cudaMalloc(&cpym1.devPtr, sizeof(float) * cpym1.size));
    CHECK_CUDA(cudaMemcpy(cpym1.devPtr, m1.devPtr, sizeof(float) * cpym1.size, cudaMemcpyHostToDevice));
    cpym2.width = m2.width; cpym2.height = m2.height; cpym2.size = m2.size;
    CHECK_CUDA(cudaMalloc(&cpym2.devPtr, sizeof(float) * cpym2.size));
    CHECK_CUDA(cudaMemcpy(cpym2.devPtr, m2.devPtr, sizeof(float) * cpym2.size, cudaMemcpyHostToDevice));
    device_m1.width = K << 5; device_m1.height = blockDimH << 5; device_m1.size = device_m1.width * device_m1.height;
    CHECK_CUDA(cudaMalloc(&device_m1.devPtr, sizeof(float) * device_m1.size));
    device_m2.width = blockDimW << 5; device_m2.height = K << 5; device_m2.size = device_m2.width * device_m2.height;
    CHECK_CUDA(cudaMalloc(&device_m2.devPtr, sizeof(float) * device_m2.size));

    PadMatrixKernel << <m1Dim, 1024 >> > (device_m1, cpym1);
    PadMatrixKernel << <m2Dim, 1024 >> > (device_m2, cpym2);

    CHECK_CUDA(cudaFree(cpym1.devPtr));
    CHECK_CUDA(cudaFree(cpym2.devPtr));

    device_m3.width = device_m2.width; device_m3.height = device_m1.height; device_m3.size = device_m3.width * device_m3.height;
    CHECK_CUDA(cudaMalloc(&device_m3.devPtr, sizeof(float) * device_m3.size));
    device_m4.width = m3->width; device_m4.height = m3->height; device_m4.size = device_m4.width * device_m4.height;
    CHECK_CUDA(cudaMalloc(&device_m4.devPtr, sizeof(float) * device_m4.size));

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    mulMatrixKernelV3 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, K);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapseTime, start, stop));
    ClipMatrixKernel << <gridDim, 1024 >> > (device_m3, device_m4);

    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m4.devPtr, sizeof(float) * device_m4.size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_m1.devPtr));
    CHECK_CUDA(cudaFree(device_m2.devPtr));
    CHECK_CUDA(cudaFree(device_m3.devPtr));
    CHECK_CUDA(cudaFree(device_m4.devPtr));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    printf("CUDA MULTIMATRIX ELAPSETIME : %0.5f\n", elapseTime);
}