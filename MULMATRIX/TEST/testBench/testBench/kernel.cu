
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "F:\ANOTHER\MINE\WORKPLACE\CUDA\MULMATRIX\TEST\testBench\testBench\Header.cuh"

#include <stdio.h>

void HostToDeviceMat(const MATRIX, MATRIX*);
float MulMat(const MATRIX, const MATRIX, MATRIX*, int);

int main()
{
    int width = 256, height = 256;
    float elapsetime;
    MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)),
         * m2 = (MATRIX*)malloc(sizeof(MATRIX)),
         * m3 = (MATRIX*)malloc(sizeof(MATRIX));

    width *= 2;
    height *= 2;
    InitMat(m1, width, height);
    InitMat(m2, height, width);
    InitMat(m3, height, height);

    SetMat(m3, 0);
    HostRANDSetMat(m1, 1);
    HostRANDSetMat(m2, 2);

    cudaDeviceSynchronize();
    MulMat(*m1, *m2, m3, 5);
    cudaDeviceSynchronize();
    DestroyMat(m1);
    DestroyMat(m2);
    DestroyMat(m3);

    return 0;
}

float MulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3, int index) {
    cudaEvent_t start, stop;
    float alpha = 1, beta = 0;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();

    HostToDeviceMat(m1, &device_m1);
    HostToDeviceMat(m2, &device_m2);
    HostToDeviceMat(*m3, &device_m3);

    dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    cudaDeviceSynchronize();
    mulMatrixKernelV5 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
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
    //printf("CUDA KERNEL %d MULTIMATRIX ELAPSETIME : %f\n", index, elapseTime);

    return elapseTime;
}

void HostToDeviceMat(const MATRIX src, MATRIX* dst) {
    dst->width = src.width;
    dst->height = src.height;
    int size = dst->width * dst->height;
    CHECK_CUDA(cudaMalloc(&dst->devPtr, sizeof(float) * size));
    CHECK_CUDA(cudaMemcpy(dst->devPtr, src.devPtr, sizeof(float) * size, cudaMemcpyHostToDevice));

}

