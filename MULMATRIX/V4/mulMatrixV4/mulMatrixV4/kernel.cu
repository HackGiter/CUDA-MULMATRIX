
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "cuda_texture_types.h"

#include "Header.cuh"

#include <stdio.h>

__host__ void HostToDeviceMat(const MATRIX, MATRIX*);
__host__ float MulMat(const MATRIX, const MATRIX, MATRIX*, int);
__host__ float CublasMM(const MATRIX, const MATRIX, MATRIX*);
__host__ void CompareSGEMM(float* , int);

int main()
{
   //int width = 256, height = 256;
   //float elapsetime1, elapsetime2;
   //MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)),
   //      * m2 = (MATRIX*)malloc(sizeof(MATRIX)),
   //      * m3 = (MATRIX*)malloc(sizeof(MATRIX)),
   //      * m4 = (MATRIX*)malloc(sizeof(MATRIX));

   //width *= 1;
   //height *= 1;
   //printf("Test Matrix : WIDTH: %d HEIGHT: %d\n", width, height);
   //InitMat(m1, width, height);
   //InitMat(m2, height, width);
   //InitMat(m3, height, height);
   //InitMat(m4, height, height);

   //SetMat(m3, 0);
   //SetMat(m4, 0);
   //HostRANDSetMat(m1, 1);
   //HostRANDSetMat(m2, 2);
   ////SetMat(m1, 1);
   ////SetMat(m2, 1.5);
   //warmup << <1, 1 >> > ();

   //cudaDeviceSynchronize();
   //elapsetime1 = MulMat(*m1, *m2, m3, 2);
   //cudaDeviceSynchronize();
   //elapsetime2 = CublasMM(*m1, *m2, m4);
   ////elapsetime2 = MulMat(*m1, *m2, m4, 5);
   //VerifyMat(*m3, *m4);
   ////PrintMat(*m3);
   //DestroyMat(m1);
   //DestroyMat(m2);
   //DestroyMat(m3);
   //DestroyMat(m4);
  /* cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   printf("%d %d %d %d %d\n", prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock, prop.regsPerMultiprocessor, prop.multiProcessorCount, prop.maxBlocksPerMultiProcessor);*/
   int times = 12;
   float* record = (float*)malloc(sizeof(float) * times * 2);
   CompareSGEMM(record, times);
   for (int i = 0; i < times; i++)
       printf("GFLOPS: SELF: %f, CUBLASS: %f\n", *(record + i), *(record + i + times));

   CHECK_CUDA(cudaDeviceReset());
    return 0;
}

__host__ float MulMat(const MATRIX m1, const MATRIX m2, MATRIX* m3, int index) {
    cudaEvent_t start, stop;
    float alpha = 1, beta = 0;
    float elapseTime;
    MATRIX device_m1 = MATRIX(), device_m2 = MATRIX(), device_m3 = MATRIX();

    HostToDeviceMat(m1, &device_m1);
    HostToDeviceMat(m2, &device_m2);
    HostToDeviceMat(*m3, &device_m3);
    
    if (index == 9) {
        dim3 gridDim(device_m1.height >> 7, device_m2.width >> 7);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelXI << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 8) {
        dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        //texture<float4, cudaTextureType1D, cudaReadModeElementType> texRefA;
        //texture<float4, cudaTextureType1D, cudaReadModeElementType> texRefB;
        //cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float4>();
        //cudaChannelFormatDesc channelDescB = cudaCreateChannelDesc<float4>();
        //size_t offsetB;
        //size_t offsetA;
        //cudaBindTexture(&offsetA, &texRefA, device_m1.devPtr, &channelDescA, sizeof(device_m1.devPtr));
        //cudaBindTexture(&offsetB, &texRefB, device_m2.devPtr, &channelDescB, sizeof(device_m2.devPtr));
        struct cudaResourceDesc resDescA;
        resDescA.resType = cudaResourceTypeLinear;
        resDescA.res.linear.devPtr = device_m1.devPtr;
        resDescA.res.linear.sizeInBytes = sizeof(float) * device_m1.width * device_m1.height;
        resDescA.res.linear.desc = cudaCreateChannelDesc<float4>();
        struct cudaTextureDesc texDescA = {};
        texDescA.readMode = cudaReadModeElementType;
        cudaTextureObject_t texA;
        cudaCreateTextureObject(&texA, &resDescA, &texDescA, NULL);
        struct cudaResourceDesc resDescB;
        resDescB.resType = cudaResourceTypeLinear;
        resDescB.res.linear.devPtr = device_m2.devPtr;
        resDescB.res.linear.sizeInBytes = sizeof(float) * device_m2.width * device_m2.height;
        resDescB.res.linear.desc = cudaCreateChannelDesc<float4>();
        struct cudaTextureDesc texDescB = {};
        texDescB.readMode = cudaReadModeElementType;
        cudaTextureObject_t texB;
        cudaCreateTextureObject(&texB, &resDescB, &texDescB, NULL);
        cudaDeviceSynchronize();
        mulMatrixKernelTV << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, texA, texB, beta, device_m3.devPtr);
        //mulMatrixKernelX << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        //mulMatrixKernelSV << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 0) {
        dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV0 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 1) {
        dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV1 << <gridDim, 128 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 2) {
        dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV2 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 3) {
        dim3 gridDim(device_m1.height >> 6, device_m2.width >> 6);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV3 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 4) {
        dim3 gridDim(device_m1.height >> 5, device_m2.width >> 5);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV4 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }
    else if (index == 5) {
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
    }
    else if (index == 6) {
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
    }
    else {
        dim3 gridDim(device_m1.height >> 7, device_m2.width >> 7);
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));
        cudaDeviceSynchronize();
        mulMatrixKernelV7 << <gridDim, 256 >> > (device_m3.height, device_m3.width, device_m1.width, alpha, device_m1.devPtr, device_m2.devPtr, beta, device_m3.devPtr);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(start));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }

    CHECK_CUDA(cudaEventElapsedTime(&elapseTime, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_m1.devPtr));
    CHECK_CUDA(cudaFree(device_m2.devPtr));
    CHECK_CUDA(cudaMemcpy(m3->devPtr, device_m3.devPtr, sizeof(float) * (m3->width * m3->height), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_m3.devPtr));
    printf("CUDA KERNEL %d MULTIMATRIX ELAPSETIME : %f\n", index, elapseTime);

    return elapseTime;
}

__host__ float CublasMM(const MATRIX m1, const MATRIX m2, MATRIX* m3) {
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
    printf("CUDA CUBLAS MULTIMATRIX ELAPSETIME : %f\n", elapseTime);
    return elapseTime;
}

__host__ void HostToDeviceMat(const MATRIX src, MATRIX* dst) {
    dst->width = src.width;
    dst->height = src.height;
    int size = dst->width * dst->height;
    CHECK_CUDA(cudaMalloc(&dst->devPtr, sizeof(float) * size));
    CHECK_CUDA(cudaMemcpy(dst->devPtr, src.devPtr, sizeof(float) * size, cudaMemcpyHostToDevice));
}

__host__ void CompareSGEMM(float* record, int times) {
    int width = 512, height = 512;
    float elapsetime1, elapsetime2, elapsetime3;
    MATRIX* m1 = (MATRIX*)malloc(sizeof(MATRIX)),
        * m2 = (MATRIX*)malloc(sizeof(MATRIX)),
        * m3 = (MATRIX*)malloc(sizeof(MATRIX)),
        * m4 = (MATRIX*)malloc(sizeof(MATRIX));

    for (int i = 0; i < times; i++) {
        width += 512;
        height += 512;
        printf("Test Matrix : WIDTH: %d HEIGHT: %d\n", width, height);
        InitMat(m1, width, height);
        InitMat(m2, height, width);
        InitMat(m3, height, height);
        InitMat(m4, height, height);

        SetMat(m3, 0);
        SetMat(m4, 0);
        HostRANDSetMat(m1, 1);
        HostRANDSetMat(m2, 2);

        warmup << <1, 1 >> > ();

        cudaDeviceSynchronize();
        elapsetime1 = MulMat(*m1, *m2, m3, 9);
        cudaDeviceSynchronize();
        elapsetime3 = CublasMM(*m1, *m2, m4);
        elapsetime2 = MulMat(*m1, *m2, m3, 7);
        VerifyMat(*m3, *m4);

        DestroyMat(m1);
        DestroyMat(m2);
        DestroyMat(m3);
        DestroyMat(m4);

        *(record + i) = (double)2 * (1e-9) * width * width * height / (elapsetime1 / 1000);
        *(record + i + times) = (double)2 * (1e-9) * width * width * height / (elapsetime3 / 1000);
    }

}
