
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <crt/device_functions.h>

#include "MATRIX.cuh"
#include "util.cuh"


#ifndef BLOCKSPERSM

#define BLOCKSPERSM         32
#define THREADSPERBLOCK     32
#define WARPSIZE            32
#define SUBBLOCKSIZE        4
#define ERROR_EXIT          -1

#endif

#define POS(ptr, x, y, width)   (ptr + y * width + x)
#define S(A, i, j) (A + (j << 5) + i)
#define INDEX(i, j) i << 4 + j

__global__ void warmup();
__global__ void RandInitMatrixKernel(MATRIX, unsigned int);
__global__ void PadMatrixKernel(MATRIX, const MATRIX);
__global__ void ClipMatrixKernel(const MATRIX, MATRIX);
__global__ void mulMatrixKernelV1(const MATRIX, const MATRIX, MATRIX);
__global__ void mulMatrixKernelV2(const MATRIX, const MATRIX, MATRIX);
__global__ void shareMulMatKernelV1(const MATRIX, const MATRIX, MATRIX, const int k, const int remain);
__global__ void shareMulMatKernelV2(const MATRIX, const MATRIX, MATRIX, const int k);

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
    //printf("%d %d %d\n", blockIdx.x, threadIdx.x, threadIdx.y);
}

__global__ void PadMatrixKernel(MATRIX dst, const MATRIX src) {
    int x = (blockIdx.x << 5) + threadIdx.x;
    int y = (blockIdx.y << 5) + threadIdx.y;
    *(dst.devPtr + y * dst.width + x) = 0;
    if (x < src.width && y < src.height) {
        *(dst.devPtr + y * dst.width + x) = *(src.devPtr + y * src.width + x);
    }
}

__global__ void ClipMatrixKernel(const MATRIX src, MATRIX dst) {
    int x = (blockIdx.x << 5) + threadIdx.x;
    int y = (blockIdx.y << 5) + threadIdx.y;
    if (x < dst.width && y < dst.height) {
        *(dst.devPtr + y * dst.width + x) = *(src.devPtr + y * src.width + x);
    }
}

__global__ void mulMatrixKernelV1(const MATRIX m1, const MATRIX m2, MATRIX m3) {
    float subBlockMatrix1[SUBBLOCKSIZE];
    float subBlockMatrix2[2][SUBBLOCKSIZE];
    float out3[2] = { 0, 0 };

    float* pos1, * pos2, * pos3;
    int row, column;
    const int ws = m2.width / (THREADSPERBLOCK * 2);
    const int hs = m1.height / BLOCKSPERSM;
    const int st = m1.width / SUBBLOCKSIZE;
    row = threadIdx.x * ws * 2;
    column = blockIdx.x * hs;
    const int remains = m1.width - st * SUBBLOCKSIZE;
    const int left = m2.width - ws * THREADSPERBLOCK * 2;
    const int choose = m1.height - hs * BLOCKSPERSM;

    for (int k = 0; k < hs; k++) {                          // every blocks take responsibility of the numbers of columns
        pos3 = m3.devPtr + row + (column + k) * m3.width;
        for (int t = 0; t < ws; t++) {                      // every threads take responsibility of the numbers of rows
            pos1 = m1.devPtr + (column + k) * m1.width;
            pos2 = m2.devPtr + row + t * 2;
            out3[0] = 0;
            out3[1] = 0;
            for (int i = 0; i < st; i++) {                  // the numbers of subblocks
                load_glm_to_reg(subBlockMatrix1, pos1, SUBBLOCKSIZE);
                load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, SUBBLOCKSIZE, m2.width);
                load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, SUBBLOCKSIZE, m2.width);
                reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], SUBBLOCKSIZE);
                reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], SUBBLOCKSIZE);
                pos1 += SUBBLOCKSIZE;
                pos2 += SUBBLOCKSIZE * m2.width;
            }
            load_glm_to_reg(subBlockMatrix1, pos1, remains);
            load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, remains, m2.width);
            load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, remains, m2.width);
            reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], remains);
            reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], remains);
            pos3[0] = out3[0];
            pos3[1] = out3[1];
            pos3 += 2;

        }
        if (threadIdx.x < (left + 1) / 2) {
            pos3 = m3.devPtr + ws * 2 * THREADSPERBLOCK + threadIdx.x * 2 + (column + k) * m3.width;
            pos1 = m1.devPtr + (column + k) * m1.width;
            pos2 = m2.devPtr + ws * 2 * THREADSPERBLOCK + threadIdx.x * 2;
            out3[0] = 0;
            out3[1] = 0;
            for (int i = 0; i < st; i++) {  // the numbers of subblocks
                load_glm_to_reg(subBlockMatrix1, pos1, SUBBLOCKSIZE);
                load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, SUBBLOCKSIZE, m2.width);
                if ((threadIdx.x + 1) * 2 <= left)
                    load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, SUBBLOCKSIZE, m2.width);
                reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], SUBBLOCKSIZE);
                if ((threadIdx.x + 1) * 2 <= left)
                    reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], SUBBLOCKSIZE);
                pos1 += SUBBLOCKSIZE;
                pos2 += SUBBLOCKSIZE * m2.width;
            }
            load_glm_to_reg(subBlockMatrix1, pos1, remains);
            load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, remains, m2.width);
            if ((threadIdx.x + 1) * 2 <= left)
                load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, remains, m2.width);
            reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], remains);
            if ((threadIdx.x + 1) * 2 <= left)
                reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], remains);
            pos3[0] = out3[0];
            if ((threadIdx.x + 1) * 2 <= left)
                pos3[1] = out3[1];
        }

    }

    if (blockIdx.x < choose) {
        column = hs * BLOCKSPERSM + blockIdx.x;
        pos3 = m3.devPtr + row + column * m3.width;
        for (int t = 0; t < ws; t++) {
            pos1 = m1.devPtr + column * m1.width;
            pos2 = m2.devPtr + row + t * 2;
            out3[0] = 0;
            out3[1] = 0;
            for (int i = 0; i < st; i++) {
                load_glm_to_reg(subBlockMatrix1, pos1, SUBBLOCKSIZE);
                load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, SUBBLOCKSIZE, m2.width);
                load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, SUBBLOCKSIZE, m2.width);
                reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], SUBBLOCKSIZE);
                reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], SUBBLOCKSIZE);
                pos1 += SUBBLOCKSIZE;
                pos2 += SUBBLOCKSIZE * m2.width;
            }
            load_glm_to_reg(subBlockMatrix1, pos1, remains);
            load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, remains, m2.width);
            load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, remains, m2.width);
            reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], remains);
            reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], remains);
            pos3[0] = out3[0];
            pos3[1] = out3[1];
            pos3 += 2;
        }
        if (threadIdx.x < (left + 1) / 2) {
            pos3 = m3.devPtr + ws * 2 * THREADSPERBLOCK + threadIdx.x * 2 + column * m3.width;
            pos1 = m1.devPtr + column * m1.width;
            pos2 = m2.devPtr + ws * 2 * THREADSPERBLOCK + threadIdx.x * 2;
            out3[0] = 0;
            out3[1] = 0;
            for (int i = 0; i < st; i++) {  // the numbers of subblocks
                load_glm_to_reg(subBlockMatrix1, pos1, SUBBLOCKSIZE);
                load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, SUBBLOCKSIZE, m2.width);
                if ((threadIdx.x + 1) * 2 <= left)
                    load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, SUBBLOCKSIZE, m2.width);
                reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], SUBBLOCKSIZE);
                if ((threadIdx.x + 1) * 2 <= left)
                    reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], SUBBLOCKSIZE);
                pos1 += SUBBLOCKSIZE;
                pos2 += SUBBLOCKSIZE * m2.width;
            }
            load_glm_to_reg(subBlockMatrix1, pos1, remains);
            load_glm_to_reg_transpose(subBlockMatrix2[0], pos2, remains, m2.width);
            if ((threadIdx.x + 1) * 2 <= left)
                load_glm_to_reg_transpose(subBlockMatrix2[1], pos2 + 1, remains, m2.width);
            reg_xmul_reg(out3, subBlockMatrix1, subBlockMatrix2[0], remains);
            if ((threadIdx.x + 1) * 2 <= left)
                reg_xmul_reg(out3 + 1, subBlockMatrix1, subBlockMatrix2[1], remains);
            pos3[0] = out3[0];
            if ((threadIdx.x + 1) * 2 <= left)
                pos3[1] = out3[1];
        }
    }
}

__global__ void mulMatrixKernelV2(const MATRIX m1, const MATRIX m2, MATRIX m3) {
    float SBM1[SUBBLOCKSIZE];
    float SBM2[SUBBLOCKSIZE];
    int tmp, yp, xp;
    int ws1 = m1.width / SUBBLOCKSIZE;
    int ws2 = m2.width / THREADSPERBLOCK;
    int hs = m1.height / BLOCKSPERSM;
    //int remain = m1.width - ws1 * SUBBLOCKSIZE;
    float out;

    for (int k = 0; k < hs; k++) {
        yp = k * BLOCKSPERSM + blockIdx.x;
        for (int t = 0; t < ws2; t++) {
            out = 0;
            xp = t * THREADSPERBLOCK + threadIdx.x;
            for (int i = 0; i < ws1; i++) {
                load_glm_to_reg(SBM1, POS(m1.devPtr, i * SUBBLOCKSIZE, yp, m1.width), SUBBLOCKSIZE);
                load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, i * SUBBLOCKSIZE, m2.width), SUBBLOCKSIZE, m2.width);
#pragma unroll
                for (int j = 0; j < SUBBLOCKSIZE; j++) out += SBM1[j] * SBM2[j];
            }
            tmp = m1.width - ws1 * SUBBLOCKSIZE;
            load_glm_to_reg(SBM1, POS(m1.devPtr, ws1 * SUBBLOCKSIZE, yp, m1.width), tmp);
            load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, ws1 * SUBBLOCKSIZE, m2.width), tmp, m2.width);
            for (int j = 0; j < tmp; j++) out += SBM1[j] * SBM2[j];
            *(m3.devPtr + yp * m3.width + xp) = out;
        }
        tmp = m2.width - ws2 * THREADSPERBLOCK;
        if (threadIdx.x < tmp) {
            out = 0;
            xp = ws2 * THREADSPERBLOCK + threadIdx.x;
            for (int i = 0; i < ws1; i++) {
                load_glm_to_reg(SBM1, POS(m1.devPtr, i * SUBBLOCKSIZE, yp, m1.width), SUBBLOCKSIZE);
                load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, i * SUBBLOCKSIZE, m2.width), SUBBLOCKSIZE, m2.width);
#pragma unroll
                for (int j = 0; j < SUBBLOCKSIZE; j++) out += SBM1[j] * SBM2[j];
            }
            tmp = m1.width - ws1 * SUBBLOCKSIZE;
            load_glm_to_reg(SBM1, POS(m1.devPtr, ws1 * SUBBLOCKSIZE, yp, m1.width), tmp);
            load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, ws1 * SUBBLOCKSIZE, m2.width), tmp, m2.width);
            for (int j = 0; j < tmp; j++) out += SBM1[j] * SBM2[j];
            *(m3.devPtr + yp * m3.width + xp) = out;
        }
    }
    tmp = m1.height - hs * BLOCKSPERSM;
    if (blockIdx.x < tmp) {
        yp = hs * BLOCKSPERSM + blockIdx.x;
        for (int t = 0; t < ws2; t++) {
            out = 0;
            xp = t * THREADSPERBLOCK + threadIdx.x;
            for (int i = 0; i < ws1; i++) {
                load_glm_to_reg(SBM1, POS(m1.devPtr, i * SUBBLOCKSIZE, yp, m1.width), SUBBLOCKSIZE);
                load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, i * SUBBLOCKSIZE, m2.width), SUBBLOCKSIZE, m2.width);
#pragma unroll
                for (int j = 0; j < SUBBLOCKSIZE; j++) out += SBM1[j] * SBM2[j];
            }
            tmp = m1.width - ws1 * SUBBLOCKSIZE;
            load_glm_to_reg(SBM1, POS(m1.devPtr, ws1 * SUBBLOCKSIZE, yp, m1.width), tmp);
            load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, ws1 * SUBBLOCKSIZE, m2.width), tmp, m2.width);
            for (int j = 0; j < tmp; j++) out += SBM1[j] * SBM2[j];
            *(m3.devPtr + yp * m3.width + xp) = out;
        }
        tmp = m2.width - ws2 * THREADSPERBLOCK;
        if (threadIdx.x < tmp) {
            out = 0;
            xp = ws2 * THREADSPERBLOCK + threadIdx.x;
            for (int i = 0; i < ws1; i++) {
                load_glm_to_reg(SBM1, POS(m1.devPtr, i * SUBBLOCKSIZE, yp, m1.width), SUBBLOCKSIZE);
                load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, i * SUBBLOCKSIZE, m2.width), SUBBLOCKSIZE, m2.width);
#pragma unroll
                for (int j = 0; j < SUBBLOCKSIZE; j++) out += SBM1[j] * SBM2[j];
            }
            tmp = m1.width - ws1 * SUBBLOCKSIZE;
            load_glm_to_reg(SBM1, POS(m1.devPtr, ws1 * SUBBLOCKSIZE, yp, m1.width), tmp);
            load_glm_to_reg_transpose(SBM2, POS(m2.devPtr, xp, ws1 * SUBBLOCKSIZE, m2.width), tmp, m2.width);
            for (int j = 0; j < tmp; j++) out += SBM1[j] * SBM2[j];
            *(m3.devPtr + yp * m3.width + xp) = out;
        }
    }
} 

__global__ void shareMulMatKernelV1(const MATRIX m1, const MATRIX m2, MATRIX m3, const int k, const int remain) {
    __shared__ float sa[THREADSPERBLOCK * THREADSPERBLOCK];
    __shared__ float sb[THREADSPERBLOCK * THREADSPERBLOCK];
    
    int offsetX = blockIdx.x << 5, offsetY = threadIdx.y + (blockIdx.y << 5), offset = m2.width << 5, pos;
    float* ldmX = m1.devPtr + offsetY * m1.width + threadIdx.x;
    float* ldmY = m2.devPtr + threadIdx.x * m2.width + offsetX + threadIdx.y;
    float x, y, tmp = 0;

#pragma unroll
    for (int i = 0; i < k; i++) {
        *S(sa, threadIdx.x, threadIdx.y) = *ldmX;
        *S(sb, threadIdx.y, threadIdx.x) = *ldmY;
        ldmX += THREADSPERBLOCK;
        ldmY += offset;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < THREADSPERBLOCK; j++) {
            pos = (threadIdx.y + j) % THREADSPERBLOCK;
            x = *S(sa, pos, threadIdx.y);
            y = *S(sb, threadIdx.x, pos);
            tmp += x * y;
        }
        __syncthreads();
    }

    if (threadIdx.x < remain) {
        *S(sa, threadIdx.x, threadIdx.y) = *ldmX;
        *S(sb, threadIdx.y, threadIdx.x) = *ldmY;
    }
    __syncthreads();

    for (int j = 0; j < remain; j++) {
        pos = (threadIdx.y + j) % remain;
        x = *S(sa, pos, threadIdx.y);
        y = *S(sb, threadIdx.x, pos);
        tmp += x * y;
    }
    __syncthreads();
    

    *(m3.devPtr + offsetX + offsetY * m3.width + threadIdx.x) = tmp;
}

__global__ void shareMulMatKernelV2(const MATRIX m1, const MATRIX m2, MATRIX m3, const int k) {
    __shared__ float sa[THREADSPERBLOCK * THREADSPERBLOCK];
    __shared__ float sb[THREADSPERBLOCK * THREADSPERBLOCK];

    int offsetX = blockIdx.x << 5, offsetY = threadIdx.y + (blockIdx.y << 5), offset = m2.width << 5, pos;
    float* ldmX = m1.devPtr + offsetY * m1.width + threadIdx.x;
    float* ldmY = m2.devPtr + threadIdx.y * m2.width + offsetX + threadIdx.x;
    float x, y, tmp = 0;

#pragma unroll
    for (int i = 0; i < k; i++) {
        *S(sa, threadIdx.x, threadIdx.y) = *ldmX;
        *S(sb, threadIdx.x, threadIdx.y) = *ldmY;
        ldmX += THREADSPERBLOCK;
        ldmY += offset;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < THREADSPERBLOCK; j++) {
            pos = (threadIdx.y + j) % THREADSPERBLOCK;
            x = *S(sa, pos, threadIdx.y);
            y = *S(sb, threadIdx.x, pos);
            tmp += x * y;
        }
        __syncthreads();
    }

    *(m3.devPtr + offsetX + offsetY * m3.width + threadIdx.x) = tmp;
}

__host__ void CUDARandInitMat(MATRIX* m, unsigned int seed) {
    MATRIX device_m = MATRIX();

    device_m.width = m->width; device_m.height = m->height; device_m.size = m->size;
    assert(cudaSuccess == cudaMalloc(&device_m.devPtr, sizeof(float) * device_m.size));

    RandInitMatrixKernel << <BLOCKSPERSM, THREADSPERBLOCK >> > (device_m, seed);

    assert(cudaSuccess == cudaMemcpy(m->devPtr, device_m.devPtr, sizeof(float) * m->size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFree(device_m.devPtr));
}

__global__ void RandInitMatrixKernel(MATRIX m, unsigned int seed) {
    //int blockSize = m.size / BLOCKSPERSM;
    //int remains = m.size - blockSize * BLOCKSPERSM;
    //int index = blockIdx.x;
    //srand(seed);
    //for (int i = 0; i < blockSize; i++) {
    //    if (i % THREADSPERBLOCK == threadIdx.x) {
    //        *(m.devPtr + index + i) = (float)rand() / 1000;
    //    }
    //}
    //if (blockIdx.x < remains && threadIdx.x == 0) {
    //    *(m.devPtr + blockSize * BLOCKSPERSM + blockIdx.x) = (float)rand() / 1000;
    //}

}

