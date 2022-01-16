
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas.h"
#include <crt/device_functions.h>

#include "MATRIX.cuh"
#include "util.cuh"

#define THREADSPERBLOCK     256
#define WARPSIZE            32
#define SUBBLOCKSIZE        32
#define ERROR_EXIT          -1
#define STEP                4

#define POS(ptr, x, y, width)   (ptr + y * width + x)
#define S(A, i, j) (A + (j << 5) + i)
#define INDEX(i, j) i << 4 + j

__global__ void warmup();
__global__ void PadMatrixKernel(MATRIX, const MATRIX);
__global__ void ClipMatrixKernel(const MATRIX, MATRIX);
__global__ void mulMatrixKernelV3(const MATRIX, const MATRIX, MATRIX, const int K);
__global__ void mulMatrixKernelV4(const MATRIX, const MATRIX, MATRIX, const int K);

/* Check the return value of CUDA Runtime API */
#define CHECK_CUDA(err) do{\
    if((err) != cudaSuccess){\
        fprintf(stderr, "CUDA Runtime API error %d at file %s line %d: %s.\n",\
                               (int)(err), __FILE__, __LINE__, cudaGetErrorString((err)));\
        exit(ERROR_EXIT);\
    }}while(0)

/* Check the return value of CURAND api. */
#define CHECK_CURAND(err) do{\
    if( (err) != CURAND_STATUS_SUCCESS ){\
        fprintf(stderr, "CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
	exit(ERROR_EXIT);\
    }}while(0)


__global__ void warmup() {
}

__global__ void PadMatrixKernel(MATRIX dst, const MATRIX src) {
    int x = (blockIdx.x << 5) + (threadIdx.x % 32);
    int y = (blockIdx.y << 5) + (threadIdx.x / 32);
    float tmp = 0;
    if (x < src.width && y < src.height) {
        tmp = *(src.devPtr + y * src.width + x);
    }
    *(dst.devPtr + y * dst.width + x) = tmp;

}

__global__ void ClipMatrixKernel(const MATRIX src, MATRIX dst) {
    int x = (blockIdx.x << 5) + (threadIdx.x % 32);
    int y = (blockIdx.y << 5) + (threadIdx.x / 32);
    if (x < dst.width && y < dst.height) {
        *(dst.devPtr + y * dst.width + x) = *(src.devPtr + y * src.width + x);
    }
}

__global__ void mulMatrixKernelV3(const MATRIX m1, const MATRIX m2, MATRIX m3, const int K) {
    __shared__ float sa[SUBBLOCKSIZE][SUBBLOCKSIZE];
    __shared__ float sb[SUBBLOCKSIZE][SUBBLOCKSIZE];
    float outcome[4] = {0, 0, 0, 0}, x, y1, y2, y3, y4;

    int step = (threadIdx.x << 2), pos;
    float* ldmX = m1.devPtr + (threadIdx.y + (blockIdx.y << 5)) * m1.width, * ldmY = m2.devPtr + threadIdx.y * m2.width + (blockIdx.x << 5);
    int r1 = (threadIdx.y & 3) + step, r2 = ((threadIdx.y + 1) & 3) + step, r3 = ((threadIdx.y + 2) & 3) + step, r4 = ((threadIdx.y + 3) & 3) + step;
    float* sma = sa[threadIdx.y], * smb = sb[threadIdx.y];
    int offset = m2.width << 5;

    for (int i = 0; i < K; i++) {
        sma[r1] = *(ldmX + r1);
        sma[r2] = *(ldmX + r2);
        sma[r3] = *(ldmX + r3);
        sma[r4] = *(ldmX + r4);
        smb[r1] = *(ldmY + r1);
        smb[r2] = *(ldmY + r2);
        smb[r3] = *(ldmY + r3);
        smb[r4] = *(ldmY + r4);
        ldmX += SUBBLOCKSIZE;
        ldmY += offset;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < SUBBLOCKSIZE; j++) {
            pos = ((threadIdx.y + j) & 31);
            smb = sb[pos];
            x = sma[pos];
            y1 = smb[r1];
            y2 = smb[r2];
            y3 = smb[r3];
            y4 = smb[r4];
            outcome[0] += y1 * x;
            outcome[1] += y2 * x;
            outcome[2] += y3 * x;
            outcome[3] += y4 * x;
        }
        smb = sb[threadIdx.y];
        __syncthreads();
    }
    pos = (threadIdx.y + (blockIdx.y << 5)) * m3.width + (blockIdx.x << 5);
    *(m3.devPtr + pos + r1) = outcome[0];
    *(m3.devPtr + pos + r2) = outcome[1];
    *(m3.devPtr + pos + r3) = outcome[2];
    *(m3.devPtr + pos  + r4) = outcome[3];
}

__global__ void mulMatrixKernelV4(const MATRIX m1, const MATRIX m2, MATRIX m3, const int K) {
    __shared__ float sa[SUBBLOCKSIZE][SUBBLOCKSIZE];
    __shared__ float sb[SUBBLOCKSIZE][SUBBLOCKSIZE];
    float4 outcome = { 0, 0, 0, 0 }, y, ldmX4, ldmY4;
    float x;

    int step = (threadIdx.x << 2), pos, offset = m2.width << 5;
    float* ldmX = m1.devPtr + (threadIdx.y + (blockIdx.y << 5)) * m1.width + step, * ldmY = m2.devPtr + threadIdx.y * m2.width + (blockIdx.x << 5) + step;
    //int r1 = (threadIdx.y & 3) + offset, r2 = ((threadIdx.y + 1) & 3) + offset, r3 = ((threadIdx.y + 2) & 3) + offset, r4 = ((threadIdx.y + 3) & 3) + offset;
    //int r1 = (threadIdx.y & 3), r2 = (threadIdx.y + 1) & 3, r3 = (threadIdx.y + 2) & 3, r4 = (threadIdx.y + 3) & 3;
    float* sma = sa[threadIdx.y], * smb = sb[threadIdx.y];
    float* tmpb;

    for (int i = 0; i < K; i++) {
        ldmX4 = *((float4*)ldmX);
        ldmY4 = *((float4*)ldmY);
        *((float4*)(sma + step)) = ldmX4;
        *((float4*)(smb + step)) = ldmY4;
        ldmX += SUBBLOCKSIZE;
        ldmY += offset;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < SUBBLOCKSIZE; j++) {
            pos = ((threadIdx.y + j) & 31);
            x = sma[pos];
            y = *((float4*)(sb[pos]));
            outcome.x += y.x * x;
            outcome.y += y.y * x;
            outcome.z += y.z * x;
            outcome.w += y.w * x;
        }
        __syncthreads();
    }
    pos = (threadIdx.y + (blockIdx.y << 5)) * m3.width + (blockIdx.x << 5) + step;
    *(float4 *)(m3.devPtr + pos) = outcome;

}