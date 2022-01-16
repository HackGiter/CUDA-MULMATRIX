
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <cublas_v2.h>
#include <crt/device_functions.h>

#include "MATRIX.cuh"
#include "util.cuh"

#define COORDGM(A, i, j, m)       (A + j * m + i)
#define COORDGMJ(A, j, m)         (A + j * m)
#define COORDGMI(A, i)            (A + i)
#define COORDSM(S, i, j, b)          (S + i + (j << b))
#define COORDSMJ(S, j, b)            (S + (j << b))
#define COORDSMI(S, i)            (S + i)
#define VECSCALE(VO, VY, a)         \
        VO.x += (double)a * VY.x;\
        VO.y += (double)a * VY.y;\
        VO.z += (double)a * VY.z;\
        VO.w += (double)a * VY.w;
#define VCPLUSSCAL(CO, CV, alpha, beta)\
        CO.x = alpha * CV.x + beta * CO.x;\
        CO.y = alpha * CV.y + beta * CO.y;\
        CO.z = alpha * CV.z + beta * CO.z;\
        CO.w = alpha * CV.w + beta * CO.w;\

#define ERROR_EXIT              -1

__global__ void warmup();
__global__ void mulMatrixKernelV5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);

#define CHECK_CUDA(err) do{\
    if((err) != cudaSuccess){\
        fprintf(stderr, "CUDA Runtime API error %d at file %s line %d: %s.\n",\
                               (int)(err), __FILE__, __LINE__, cudaGetErrorString((err)));\
        exit(ERROR_EXIT);\
    }}while(0)

#define CHECK_CURAND(err) do{\
    if( (err) != CURAND_STATUS_SUCCESS ){\
        fprintf(stderr, "CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
	exit(ERROR_EXIT);\
    }}while(0)


__global__ void warmup() {
}

/*
M: number of rows in C
N: number of columns in C
K: number of columns in A / number of rows in B
*/
__global__ void mulMatrixKernelV5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) { 
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[4], CO[4], AV, BV;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int coordax = (tx & 15) << 2, coorday = tx >> 4, coordbx = (tx & 3) << 2, coordby = tx >> 2;

    float* ldisma = COORDSM(sa, coordax, coorday, 6), 
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordax),
         * ldosmb = COORDSMI(sb, (coorday << 2));
    float* ldmA = COORDGM(A, (bx + coordax), coorday, M),
         * ldmB = COORDGM(B, coordbx, (coordby + by), K),
         * ldmC = COORDGM(C, (bx + coordax), ((coorday << 2) + by), M);
    int offset = M << 4;

    for (int i = 0; i < K; i += 16) {
        AV = *((float4*)ldmA);
        BV = *((float4*)ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        ldmA += offset;
        ldmB += 16;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j ++) {
            AV = *((float4*)COORDSMJ(ldosma, j, 6));
            BV = *((float4*)COORDSMJ(ldosmb, j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(ldmC, 0, M));
    CO[1] = *((float4*)COORDGMJ(ldmC, 1, M));
    CO[2] = *((float4*)COORDGMJ(ldmC, 2, M));
    CO[3] = *((float4*)COORDGMJ(ldmC, 3, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(ldmC, 0, M)) = CO[0];
    *((float4*)COORDGMJ(ldmC, 1, M)) = CO[1];
    *((float4*)COORDGMJ(ldmC, 2, M)) = CO[2];
    *((float4*)COORDGMJ(ldmC, 3, M)) = CO[3];
}

__global__ void mulMatrixKernelV6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[16], AV, BV, AVE, BVE;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 7, by = blockIdx.y << 7;
    int wid = tx >> 5, bid = tx & 31;
    int coordax = (tx & 31) << 2, coorday = tx >> 5,
        coordbx = (tx & 1) << 2, coordby = tx >> 1,
        coordcx = ((bid & 3) << 3) + ((wid & 3) << 5),
        coordcy = ((bid / 4) << 3) + ((wid / 4) << 6);

    float* ldisma = COORDSM(sa, coordax, coorday, 7),
         * ldismb = COORDSM(sb, coordby, coordbx, 7),
         * ldosma = COORDSMI(sa, coordcx),
         * ldosmb = COORDSMI(sb, coordcy);
    float* ldmA = COORDGM(A, (bx + coordax), coorday, M),
         * ldmB = COORDGM(B, coordbx, (by + coordby), K),
         * ldmC = COORDGM(C, (bx + coordcx), (by + coordcy), M);
    int offset = M << 3;

    for (int i = 0; i < K; i += 8) {
        AV = *((float4*)ldmA);
        BV = *((float4*)ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 7) = BV.x;
        *COORDSMJ(ldismb, 1, 7) = BV.y;
        *COORDSMJ(ldismb, 2, 7) = BV.z;
        *COORDSMJ(ldismb, 3, 7) = BV.w;
        ldmA += offset;
        ldmB += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            AV = *((float4*)COORDSMJ(ldosma, j, 7));
            BV = *((float4*)COORDSMJ(ldosmb, j, 7));
            AVE = *((float4*)COORDSM(ldosma, 4, j, 7));
            BVE = *((float4*)COORDSM(ldosmb, 4, j, 7));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AVE, BV.x);
            VECSCALE(CV[2], AV, BV.y);
            VECSCALE(CV[3], AVE, BV.y);
            VECSCALE(CV[4], AV, BV.z);
            VECSCALE(CV[5], AVE, BV.z);
            VECSCALE(CV[6], AV, BV.w);
            VECSCALE(CV[7], AVE, BV.w);
            VECSCALE(CV[8], AV, BVE.x);
            VECSCALE(CV[9], AVE, BVE.x);
            VECSCALE(CV[10], AV, BVE.y);
            VECSCALE(CV[11], AVE, BVE.y);
            VECSCALE(CV[12], AV, BVE.z);
            VECSCALE(CV[13], AVE, BVE.z);
            VECSCALE(CV[14], AV, BVE.w);
            VECSCALE(CV[15], AVE, BVE.w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(ldmC, 0, M));
    CO[1] = *((float4*)COORDGM(ldmC, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(ldmC, 1, M));
    CO[3] = *((float4*)COORDGM(ldmC, 4, 1, M));
    CO[4] = *((float4*)COORDGMJ(ldmC, 2, M));
    CO[5] = *((float4*)COORDGM(ldmC, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(ldmC, 3, M));
    CO[7] = *((float4*)COORDGM(ldmC, 4, 3, M));
    CO[8] = *((float4*)COORDGMJ(ldmC, 4, M));
    CO[9] = *((float4*)COORDGM(ldmC, 4, 4, M));
    CO[10] = *((float4*)COORDGMJ(ldmC, 5, M));
    CO[11] = *((float4*)COORDGM(ldmC, 4, 5, M));
    CO[12] = *((float4*)COORDGMJ(ldmC, 6, M));
    CO[13] = *((float4*)COORDGM(ldmC, 4, 6, M));
    CO[14] = *((float4*)COORDGMJ(ldmC, 7, M));
    CO[15] = *((float4*)COORDGM(ldmC, 4, 7, M));

    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    VCPLUSSCAL(CO[8], CV[8], alpha, beta);
    VCPLUSSCAL(CO[9], CV[9], alpha, beta);
    VCPLUSSCAL(CO[10], CV[10], alpha, beta);
    VCPLUSSCAL(CO[11], CV[11], alpha, beta);
    VCPLUSSCAL(CO[12], CV[12], alpha, beta);
    VCPLUSSCAL(CO[13], CV[13], alpha, beta);
    VCPLUSSCAL(CO[14], CV[14], alpha, beta);
    VCPLUSSCAL(CO[15], CV[15], alpha, beta);

    *((float4*)COORDGMJ(ldmC, 0, M)) = CO[0];
    *((float4*)COORDGM(ldmC, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(ldmC, 1, M)) = CO[2];
    *((float4*)COORDGM(ldmC, 4, 1, M)) = CO[3];
    *((float4*)COORDGMJ(ldmC, 2, M)) = CO[4];
    *((float4*)COORDGM(ldmC, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(ldmC, 3, M)) = CO[6];
    *((float4*)COORDGM(ldmC, 4, 3, M)) = CO[7];
    *((float4*)COORDGMJ(ldmC, 4, M)) = CO[8];
    *((float4*)COORDGM(ldmC, 4, 4, M)) = CO[9];
    *((float4*)COORDGMJ(ldmC, 5, M)) = CO[10];
    *((float4*)COORDGM(ldmC, 4, 5, M)) = CO[11];
    *((float4*)COORDGMJ(ldmC, 6, M)) = CO[12];
    *((float4*)COORDGM(ldmC, 4, 6, M)) = CO[13];
    *((float4*)COORDGMJ(ldmC, 7, M)) = CO[14];
    *((float4*)COORDGM(ldmC, 4, 7, M)) = CO[15];
}
