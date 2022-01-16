
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include "Header.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <time.h>

__host__ void MulMat(const MATRIX, const MATRIX, MATRIX*);
__host__ void SimMulMat(const MATRIX, const MATRIX, MATRIX*);
__host__ void MultiplyMatrix(const MATRIX, const MATRIX, MATRIX*);
__host__ void CompareMatrix(const MATRIX);

int main()
{
    int n = 4;
    int m = 4;
    int width = THREADSPERBLOCK * n;
    int height = THREADSPERBLOCK * m;
    MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)), * m2 = (MATRIX*)malloc(sizeof(MATRIX)), * m3 = (MATRIX*)malloc(sizeof(MATRIX));

    InitMat(m1, width + 2, height + 4);
    InitMat(m2, height + 4, width + 2);
    InitMat(m3, height + 4, height + 4);
    
    SetMat(m1, 1.0);
    SetMat(m2, 1.5);

    //HostRANDInitMat(m1, 1);
    //HostRANDInitMat(m2, 2);
    dim3 threadsPerBlock(2, THREADSPERBLOCK);
    warmup << <BLOCKSPERSM, threadsPerBlock >> > ();
    //HostRANDInitMat(m1, 1);
    //MulMat(*m1, *m2, m3);
    //PrintMat(*m3);
    MultiplyMatrix(*m1, *m2, m3);
    //SimMulMat(*m1, *m2, m3);
    //CUDARandInitMat(m1, 1);

    PrintMat(*m3);

    return 0;
}

__host__ void MulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3) {

    cudaEvent_t start, stop;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();

    device_m1.width = m1.width; device_m1.height = m1.height; device_m1.size = m1.size;
    assert(cudaSuccess == cudaMalloc(&device_m1.devPtr, sizeof(float) * device_m1.size));
    assert(cudaSuccess == cudaMemcpy(device_m1.devPtr, m1.devPtr, sizeof(float) * device_m1.size, cudaMemcpyHostToDevice));

    device_m2.width = m2.width; device_m2.height = m2.height; device_m2.size = m2.size;
    assert(cudaSuccess == cudaMalloc(&device_m2.devPtr, sizeof(float) * device_m2.size));
    assert(cudaSuccess == cudaMemcpy(device_m2.devPtr, m2.devPtr, sizeof(float) * device_m2.size, cudaMemcpyHostToDevice));

    device_m3.width = m3->width; device_m3.height = m3->height; device_m3.size = m3->size;
    assert(cudaSuccess == cudaMalloc(&device_m3.devPtr, sizeof(float) * device_m3.size));
    
    dim3 blockDim(THREADSPERBLOCK, THREADSPERBLOCK);
    //dim3 gridDim(CEIL_DIV(device_m1.width, THREADSPERBLOCK), CEIL_DIV(device_m1.height, THREADSPERBLOCK));
    int blockDimW = device_m1.width / THREADSPERBLOCK, blockDimH = device_m1.height / THREADSPERBLOCK, remain = device_m1.width - blockDimW * THREADSPERBLOCK;
    dim3 gridDim(blockDimW, blockDimH);
    assert(cudaSuccess == cudaEventCreate(&start));
    assert(cudaSuccess == cudaEventCreate(&stop));
    assert(cudaSuccess == cudaEventRecord(start, 0));
    //mulMatrixKernelV1 << <BLOCKSPERSM, THREADSPERBLOCK >> > (device_m1, device_m2, device_m3);
    //mulMatrixKernelV2 << < BLOCKSPERSM, THREADSPERBLOCK >> > (device_m1, device_m2, device_m3);
    //shareMulMatKernelV1 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, CEIL_DIV(device_m1.width, THREADSPERBLOCK));
    shareMulMatKernelV1 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, blockDimW, remain);

    CHECK_CUDA(cudaEventRecord(stop, 0));
    assert(cudaSuccess == cudaEventSynchronize(stop));
    assert(cudaSuccess == cudaEventElapsedTime(&elapseTime, start, stop));
    
    assert(cudaSuccess == cudaMemcpy(m3->devPtr, device_m3.devPtr, sizeof(float) * m3->size, cudaMemcpyDeviceToHost));
    assert(cudaSuccess == cudaFree(device_m1.devPtr));
    assert(cudaSuccess == cudaFree(device_m2.devPtr));
    assert(cudaSuccess == cudaFree(device_m3.devPtr));
    assert(cudaSuccess == cudaEventDestroy(start));
    assert(cudaSuccess == cudaEventDestroy(stop));
    printf("CUDA MULTIMATRIX ELAPSETIME : %0.5f\n", elapseTime);
    
}

__host__ void SimMulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();
 
    device_m1.width = m1.width; device_m1.height = m1.height; device_m1.size = m1.size;
    assert(cudaSuccess == cudaMalloc(&device_m1.devPtr, sizeof(float) * device_m1.size));
    assert(cudaSuccess == cudaMemcpy(device_m1.devPtr, m1.devPtr, sizeof(float) * device_m1.size, cudaMemcpyHostToDevice));

    device_m2.width = m2.width; device_m2.height = m2.height; device_m2.size = m2.size;
    assert(cudaSuccess == cudaMalloc(&device_m2.devPtr, sizeof(float) * device_m2.size));
    assert(cudaSuccess == cudaMemcpy(device_m2.devPtr, m2.devPtr, sizeof(float) * device_m2.size, cudaMemcpyHostToDevice));

    device_m3.width = m3->width; device_m3.height = m3->height; device_m3.size = m3->size;
    assert(cudaSuccess == cudaMalloc(&device_m3.devPtr, sizeof(float) * device_m3.size));

    dim3 blockDim(THREADSPERBLOCK, THREADSPERBLOCK);
    int blockDimW = device_m1.width / THREADSPERBLOCK, blockDimH = device_m1.height / THREADSPERBLOCK, remain = device_m1.width - blockDimW * THREADSPERBLOCK;
    //dim3 gridDim(CEIL_DIV(device_m1.width, THREADSPERBLOCK), CEIL_DIV(device_m1.height, THREADSPERBLOCK));
    //shareMulMatKernelV1 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, CEIL_DIV(device_m1.width, THREADSPERBLOCK));
    dim3 gridDim(blockDimW, blockDimH);
    shareMulMatKernelV1 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, blockDimW, remain);

    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m3.devPtr, sizeof(float) * m3->size, cudaMemcpyDeviceToHost));
    assert(cudaSuccess == cudaFree(device_m1.devPtr));
    assert(cudaSuccess == cudaFree(device_m2.devPtr));
    assert(cudaSuccess == cudaFree(device_m3.devPtr));
}

__host__ void MultiplyMatrix(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
    cudaEvent_t start, stop;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX(), device_m4 = MATRIX(), cpym1 = MATRIX(), cpym2 = MATRIX();

    int K = CEIL_DIV(m1.width, THREADSPERBLOCK), int blockDimW = CEIL_DIV(m2.width, THREADSPERBLOCK), blockDimH = CEIL_DIV(m1.height, THREADSPERBLOCK);
    dim3 m1Dim(K, blockDimH);
    dim3 m2Dim(blockDimW, K);
    dim3 gridDim(blockDimW, blockDimH);
    dim3 blockDim(THREADSPERBLOCK, THREADSPERBLOCK);

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

    PadMatrixKernel << <m1Dim, blockDim >> > (device_m1, cpym1);
    PadMatrixKernel << <m2Dim, blockDim >> > (device_m2, cpym2);

    CHECK_CUDA(cudaFree(cpym1.devPtr));
    CHECK_CUDA(cudaFree(cpym2.devPtr));

    device_m3.width = device_m2.width; device_m3.height = device_m1.height; device_m3.size = device_m3.width * device_m3.height;
    CHECK_CUDA(cudaMalloc(&device_m3.devPtr, sizeof(float) * device_m3.size));
    device_m4.width = m3->width; device_m4.height = m3->height; device_m4.size = device_m4.width * device_m4.height;
    CHECK_CUDA(cudaMalloc(&device_m4.devPtr, sizeof(float) * device_m4.size));

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    shareMulMatKernelV2 << <gridDim, blockDim >> > (device_m1, device_m2, device_m3, K);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapseTime, start, stop));
    ClipMatrixKernel << <gridDim, blockDim >> > (device_m3, device_m4);

    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m4.devPtr, sizeof(float) * device_m4.size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_m1.devPtr));
    CHECK_CUDA(cudaFree(device_m2.devPtr));
    CHECK_CUDA(cudaFree(device_m3.devPtr));
    CHECK_CUDA(cudaFree(device_m4.devPtr));

    assert(cudaSuccess == cudaEventDestroy(start));
    assert(cudaSuccess == cudaEventDestroy(stop));
    printf("CUDA MULTIMATRIX ELAPSETIME : %0.5f\n", elapseTime);
}

__host__ void CompareMatrix(const MATRIX m) {
    MATRIX cm = MATRIX();
    cm.width = m.width; cm.height = m.height; cm.size = cm.width * cm.height;
    cm.devPtr = (float*)malloc(sizeof(float) * cm.size);
    CHECK_CUDA(cudaMemcpy(cm.devPtr, m.devPtr, sizeof(float) * cm.size, cudaMemcpyDeviceToHost));
    PrintMat(cm);
}



  