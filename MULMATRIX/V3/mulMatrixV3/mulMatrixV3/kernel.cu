
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "Header.cuh"

#include <stdio.h>

__host__ void HostToDeviceMat(const MATRIX, MATRIX*);
__host__ void MulMat(const MATRIX, const MATRIX, MATRIX*);
__host__ void CublasMM(const MATRIX, const MATRIX, MATRIX*);

int main()
{
    int n = 20;
    int m = 20;
    int width = 64 * n;
    int height = 64 * m;
    MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)), 
          * m2 = (MATRIX*)malloc(sizeof(MATRIX)), 
          * m3 = (MATRIX*)malloc(sizeof(MATRIX)), 
          * m4 = (MATRIX*)malloc(sizeof(MATRIX));

    InitMat(m1, width, height);
    InitMat(m2, height, width);
    InitMat(m3, height, height);
    InitMat(m4, height, height);

    SetMat(m3, 0);
    SetMat(m4, 0);
    //SetMat(m1, 1);
    //SetMat(m2, 1.5);
    HostRANDSetMat(m1, 1);
    HostRANDSetMat(m2, 2);

    warmup << <1, 1 >> > ();
    cudaDeviceSynchronize();
    MulMat(*m1, *m2, m3);
    cudaDeviceSynchronize();
    CublasMM(*m1, *m2, m4);
    VerifyMat(*m3, *m4);

    //PrintMat(*m3);

    return 0;
}

__host__ void MulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
    cudaEvent_t start, stop;
    float alpha = 1, beta = 0;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();

    HostToDeviceMat(m1, &device_m1);
    HostToDeviceMat(m2, &device_m2);
    HostToDeviceMat(*m3, &device_m3);
    //dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
    dim3 gridDim(device_m1.height >> 7, device_m2.width >> 7);

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    cudaDeviceSynchronize();
    mulMatrixKernelV6 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapseTime, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_m1.devPtr));
    CHECK_CUDA(cudaFree(device_m2.devPtr));
    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m3.devPtr, sizeof(float) * (m3->width * m3->height), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_m3.devPtr));
    printf("CUDA SELF MULTIMATRIX ELAPSETIME : %0.9f\n", elapseTime);
}

__host__ void CublasMM(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
    cudaEvent_t start, stop;
    float elapseTime;
    cublasHandle_t err; cublasCreate(&err);
    float alpha = 1, beta = 0;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();
    HostToDeviceMat(m1, &device_m1);
    HostToDeviceMat(m2, &device_m2);
    HostToDeviceMat(*m3, &device_m3);

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    cudaDeviceSynchronize();
    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, device_m3.height, device_m3.width, device_m1.width, &alpha, device_m1.devPtr, device_m1.height, device_m2.devPtr, device_m2.height, &beta, device_m3.devPtr, device_m3.height);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapseTime, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_m1.devPtr));
    CHECK_CUDA(cudaFree(device_m2.devPtr));
    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m3.devPtr, sizeof(float) * (m3->width * m3->height), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_m3.devPtr));
    printf("CUDA CUBLAS MULTIMATRIX ELAPSETIME : %0.9f\n", elapseTime);
}

__host__ void HostToDeviceMat(const MATRIX src, MATRIX* dst) {
    dst->width = src.width;
    dst->height = src.height;
    int size = dst->width * dst->height;
    CHECK_CUDA(cudaMalloc(&dst->devPtr, sizeof(float) * size));
    CHECK_CUDA(cudaMemcpy(dst->devPtr, src.devPtr, sizeof(float) * size, cudaMemcpyHostToDevice));
}

