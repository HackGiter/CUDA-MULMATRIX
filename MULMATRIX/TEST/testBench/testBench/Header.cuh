#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include "F:\ANOTHER\MINE\WORKPLACE\CUDA\MULMATRIX\TEST\testBench\testBench\MATRIX.cuh"
#include "F:\ANOTHER\MINE\WORKPLACE\CUDA\MULMATRIX\TEST\testBench\testBench\util.cuh"

#define COORDGM(A, i, j, m)       ((A) + (j) * (m) + (i))
#define COORDGMJ(A, j, m)         ((A) + (j) * (m))
#define COORDGMI(A, i)            ((A) + (i))
#define COORDSM(S, i, j, b)       ((S) + ((j) << (b)) + (i))
#define COORDSMJ(S, j, b)         ((S) + ((j) << (b)))
#define COORDSMI(S, i)            ((S) + (i))
#define MEM2REG(R, M)              R = *((float4*)(M))
#define REG2MEM(M, R)              *((float4*)(M)) = R  
#define REG2REG(D, S)             (D) = (S);
#define VECSCALE(VO, VY, a)\
        (VO).x += (double)a * (VY).x;\
        (VO).y += (double)a * (VY).y;\
        (VO).z += (double)a * (VY).z;\
        (VO).w += (double)a * (VY).w;
#define VCPLUSSCAL(CO, CV, alpha, beta)\
        (CO).x = alpha * (CV).x + beta * (CO).x;\
        (CO).y = alpha * (CV).y + beta * (CO).y;\
        (CO).z = alpha * (CV).z + beta * (CO).z;\
        (CO).w = alpha * (CV).w + beta * (CO).w;\

#define ERROR_EXIT              -1

__global__ void warmup();
__global__ void mulMatrixKernelV7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);

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
__global__ void mulMatrixKernelV1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[512];
    __shared__ float sb[512];
    float4 CV[8], CO[4], AV, BV, EBV;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int wid = tx >> 5, bid = tx & 31;
    int coordax = (bid & 15) << 2, coorday = tx >> 4,
        coordbx = (bid & 1) << 2, coordby = tx / 2,
        coordcx = ((bid & 7) << 2) + ((wid & 1) << 5),
        coordcy = ((bid / 8) << 3) + ((wid / 2) << 5);
    float* ldisma = COORDSM(sa, coordax, coorday, 6),
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordcx),
         * ldosmb = COORDSMI(sb, coordcy);
    float* ldmA = COORDGM(A, coordax + bx, coorday, M),
         * ldmB = COORDGM(B, coordbx, coordby + by, K),
         * ldmC = COORDGM(C, coordcx + bx, coordcy + by, M);
    int offset = M << 3;

    for (int i = 0; i < K; i += 8) {
        MEM2REG(AV, ldmA);
        MEM2REG(BV, ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        ldmA += offset;
        ldmB += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MEM2REG(AV, COORDSMJ(ldosma, j, 6));
            MEM2REG(BV, COORDSMJ(ldosmb, j, 6));
            MEM2REG(EBV, COORDSMJ(ldosmb + 4, j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
            VECSCALE(CV[4], AV, EBV.x);
            VECSCALE(CV[5], AV, EBV.y);
            VECSCALE(CV[6], AV, EBV.z);
            VECSCALE(CV[7], AV, EBV.w);
        }
        __syncthreads();
    }

    MEM2REG(CO[0], COORDGMJ(ldmC, 0, M));
    MEM2REG(CO[1], COORDGMJ(ldmC, 1, M));
    MEM2REG(CO[2], COORDGMJ(ldmC, 2, M));
    MEM2REG(CO[3], COORDGMJ(ldmC, 3, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    REG2MEM(COORDGMJ(ldmC, 0, M), CO[0]);
    REG2MEM(COORDGMJ(ldmC, 1, M), CO[1]);
    REG2MEM(COORDGMJ(ldmC, 2, M), CO[2]);
    REG2MEM(COORDGMJ(ldmC, 3, M), CO[3]);
    MEM2REG(CO[0], COORDGMJ(ldmC, 4, M));
    MEM2REG(CO[1], COORDGMJ(ldmC, 5, M));
    MEM2REG(CO[2], COORDGMJ(ldmC, 6, M));
    MEM2REG(CO[3], COORDGMJ(ldmC, 7, M));
    VCPLUSSCAL(CO[0], CV[4], alpha, beta);
    VCPLUSSCAL(CO[1], CV[5], alpha, beta);
    VCPLUSSCAL(CO[2], CV[6], alpha, beta);
    VCPLUSSCAL(CO[3], CV[7], alpha, beta);
    REG2MEM(COORDGMJ(ldmC, 4, M), CO[0]);
    REG2MEM(COORDGMJ(ldmC, 5, M), CO[1]);
    REG2MEM(COORDGMJ(ldmC, 6, M), CO[2]);
    REG2MEM(COORDGMJ(ldmC, 7, M), CO[3]);
}

__global__ void mulMatrixKernelV2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[512];
    __shared__ float sb[512];
    float4 CV[8], CO[8], AV, EAV, BV, EBV, TBV;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int wid = tx >> 5, bid = tx & 31;
    int coordax = (bid & 15) << 2, coorday = tx >> 4,
        coordbx = (bid & 1) << 2, coordby = tx / 2,
        coordcx = ((bid & 7) << 2) + ((wid & 1) << 5),
        coordcy = ((bid / 8) << 3) + ((wid / 2) << 5);
    float* ldisma = COORDSM(sa, coordax, coorday, 6),
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordcx),
         * ldosmb = COORDSMI(sb, coordcy);
    float* ldmA = COORDGM(A, coordax + bx, coorday, M),
         * ldmB = COORDGM(B, coordbx, coordby + by, K),
         * ldmC = COORDGM(C, coordcx + bx, coordcy + by, M);
    int offset = M << 3;

    MEM2REG(AV, ldmA);
    MEM2REG(BV, ldmB);
    for (int i = 8; i < K; i += 8) {
        ldmA += offset;
        ldmB += 8;
        MEM2REG(EAV, ldmA);
        MEM2REG(EBV, ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MEM2REG(AV, COORDSMJ(ldosma, j, 6));
            MEM2REG(BV, COORDSMJ(ldosmb, j, 6));
            MEM2REG(TBV, COORDSMJ(ldosmb + 4, j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
            VECSCALE(CV[4], AV, TBV.x);
            VECSCALE(CV[5], AV, TBV.y);
            VECSCALE(CV[6], AV, TBV.z);
            VECSCALE(CV[7], AV, TBV.w);
        }
        REG2REG(AV, EAV);
        REG2REG(BV, EBV);
        __syncthreads();
    }
    *((float4*)ldisma) = AV;
    *COORDSMJ(ldismb, 0, 6) = BV.x;
    *COORDSMJ(ldismb, 1, 6) = BV.y;
    *COORDSMJ(ldismb, 2, 6) = BV.z;
    *COORDSMJ(ldismb, 3, 6) = BV.w;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
        MEM2REG(AV, COORDSMJ(ldosma, j, 6));
        MEM2REG(BV, COORDSMJ(ldosmb, j, 6));
        MEM2REG(TBV, COORDSMJ(ldosmb + 4, j, 6));
        VECSCALE(CV[0], AV, BV.x);
        VECSCALE(CV[1], AV, BV.y);
        VECSCALE(CV[2], AV, BV.z);
        VECSCALE(CV[3], AV, BV.w);
        VECSCALE(CV[4], AV, TBV.x);
        VECSCALE(CV[5], AV, TBV.y);
        VECSCALE(CV[6], AV, TBV.z);
        VECSCALE(CV[7], AV, TBV.w);
    }
    MEM2REG(CO[0], COORDGMJ(ldmC, 0, M));
    MEM2REG(CO[1], COORDGMJ(ldmC, 1, M));
    MEM2REG(CO[2], COORDGMJ(ldmC, 2, M));
    MEM2REG(CO[3], COORDGMJ(ldmC, 3, M));
    MEM2REG(CO[4], COORDGMJ(ldmC, 4, M));
    MEM2REG(CO[5], COORDGMJ(ldmC, 5, M));
    MEM2REG(CO[6], COORDGMJ(ldmC, 6, M));
    MEM2REG(CO[7], COORDGMJ(ldmC, 7, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    REG2MEM(COORDGMJ(ldmC, 0, M), CO[0]);
    REG2MEM(COORDGMJ(ldmC, 1, M), CO[1]);
    REG2MEM(COORDGMJ(ldmC, 2, M), CO[2]);
    REG2MEM(COORDGMJ(ldmC, 3, M), CO[3]);
    REG2MEM(COORDGMJ(ldmC, 4, M), CO[4]);
    REG2MEM(COORDGMJ(ldmC, 5, M), CO[5]);
    REG2MEM(COORDGMJ(ldmC, 6, M), CO[6]);
    REG2MEM(COORDGMJ(ldmC, 7, M), CO[7]);
}

__global__ void mulMatrixKernelV3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[4], CO[4], AV, BV, EAV, EBV;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int wid = tx >> 5, bid = tx & 31;
    int coordax = (bid & 15) << 2, coorday = tx >> 4,
        coordbx = (bid & 3) << 2, coordby = tx >> 2,
        coordcx = ((bid & 3) << 2) + ((wid & 3) << 4),
        coordcy = ((bid / 4) << 2) + ((wid / 4) << 5);
    float* ldisma = COORDSM(sa, coordax, coorday, 6),
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordcx),
         * ldosmb = COORDSMI(sb, coordcy);
    float* ldmA = COORDGM(A, coordax + bx, coorday, M),
         * ldmB = COORDGM(B, coordbx, coordby + by, K),
         * ldmC = COORDGM(C, coordcx + bx, coordcy + by, M);
    int offset = M << 4;

    MEM2REG(AV, ldmA);
    MEM2REG(BV, ldmB);
    for (int i = 16; i < K; i += 16) {
        ldmA += offset;
        ldmB += 16;
        MEM2REG(EAV, ldmA);
        MEM2REG(EBV, ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j++) {
            AV = *((float4*)COORDSMJ(ldosma, j, 6));
            BV = *((float4*)COORDSMJ(ldosmb, j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
        }
        REG2REG(AV, EAV);
        REG2REG(BV, EBV);
        __syncthreads();
    }
    *((float4*)ldisma) = AV;
    *COORDSMJ(ldismb, 0, 6) = BV.x;
    *COORDSMJ(ldismb, 1, 6) = BV.y;
    *COORDSMJ(ldismb, 2, 6) = BV.z;
    *COORDSMJ(ldismb, 3, 6) = BV.w;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; j++) {
        AV = *((float4*)COORDSMJ(ldosma, j, 6));
        BV = *((float4*)COORDSMJ(ldosmb, j, 6));
        VECSCALE(CV[0], AV, BV.x);
        VECSCALE(CV[1], AV, BV.y);
        VECSCALE(CV[2], AV, BV.z);
        VECSCALE(CV[3], AV, BV.w);
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

__global__ void mulMatrixKernelV4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV, CO, AV, BV[2];
    memset(&CV, 0, sizeof(CV));
    float x;
    int tx = threadIdx.x, bx = blockIdx.x << 5, by = blockIdx.y << 5;
    int wid = tx >> 5, bid = tx & 31;
    int coordcx = (wid << 2) + (bid / 8), coordcy = (bid & 7) << 2;
    int offset = M << 5, index, shift = (bid / 8) << 1;
    float* ldmA = COORDGM(A, (wid << 2) + bx, bid, M),
         * ldmB = COORDGM(B, wid << 2, bid + by, K),
         * ldmC = COORDGM(C, coordcx + bx, coordcy + by, M);
    float* ldisma = COORDSM(sa, bid, wid, 7), 
         * ldismb = COORDSM(sb, bid, wid, 7),
         * ldosma = COORDSMJ(sa, coordcx, 5),
         * ldosmb = COORDSMI(sb, coordcy);
    for (int i = 0; i < K; i += 32) {
        MEM2REG(AV, ldmA);
        MEM2REG(BV[0], ldmB);
        *COORDSMJ(ldisma, 0, 5) = AV.x;
        *COORDSMJ(ldisma, 1, 5) = AV.y;
        *COORDSMJ(ldisma, 2, 5) = AV.z;
        *COORDSMJ(ldisma, 3, 5) = AV.w;
        *COORDSMJ(ldismb, 0, 5) = BV[0].x;
        *COORDSMJ(ldismb, 1, 5) = BV[0].y;
        *COORDSMJ(ldismb, 2, 5) = BV[0].z;
        *COORDSMJ(ldismb, 3, 5) = BV[0].w;
        ldmA += offset;
        ldmB += 32;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            index = ((shift + j) & 7) << 2;
            MEM2REG(BV[0], COORDSMJ(ldosmb, index, 5));
            MEM2REG(AV, COORDSMI(ldosma, index));
#pragma unroll
            for (int k = 1; k < 4; k++)
            {
                MEM2REG(BV[k & 1], COORDSMJ(ldosmb, index + k, 5));
                x = ((float*)(&AV))[k - 1];
                VECSCALE(CV, BV[(k + 1) & 1], x);
            }
            x = ((float*)(&AV))[3];
            VECSCALE(CV, BV[1], x);
        }
        __syncthreads();
    }
    CO.x = *COORDGMJ(ldmC, 0, M);
    CO.y = *COORDGMJ(ldmC, 1, M);
    CO.z = *COORDGMJ(ldmC, 2, M);
    CO.w = *COORDGMJ(ldmC, 3, M);
    VCPLUSSCAL(CO, CV, alpha, beta);
    *COORDGMJ(ldmC, 0, M) = CO.x;
    *COORDGMJ(ldmC, 1, M) = CO.y;
    *COORDGMJ(ldmC, 2, M) = CO.z;
    *COORDGMJ(ldmC, 3, M) = CO.w;
}

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
        MEM2REG(AV, ldmA);
        MEM2REG(BV, ldmB);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        ldmA += offset;
        ldmB += 16;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j++) {
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

__global__ void mulMatrixKernelV7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[16], AV[2], BV[2], EAV[2], EBV[2], NAV, NBV;
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
    float* ldmA = COORDGM(A, bx + coordax, coorday, M),
        * ldmB = COORDGM(B, coordbx, by + coordby, K),
        * ldmC = COORDGM(C, bx + coordcx, by + coordcy, M);
    float4* WA, * WB, * EWA, * EWB;
    int offset = M << 3, index;

    MEM2REG(NAV, ldmA);
    MEM2REG(NBV, ldmB);
    REG2MEM(ldisma, NAV);
    *COORDSMJ(ldismb, 0, 7) = NBV.x;
    *COORDSMJ(ldismb, 1, 7) = NBV.y;
    *COORDSMJ(ldismb, 2, 7) = NBV.z;
    *COORDSMJ(ldismb, 3, 7) = NBV.w;
    __syncthreads();
    MEM2REG(AV[0], COORDSMJ(ldosma, 0, 7));
    MEM2REG(BV[0], COORDSMJ(ldosmb, 0, 7));
    MEM2REG(EAV[0], COORDSM(ldosma, 4, 0, 7));
    MEM2REG(EBV[0], COORDSM(ldosmb, 4, 0, 7));
    for (int i = 8; i < K; i += 8) {
        ldmA += offset;
        ldmB += 8;
        MEM2REG(NAV, ldmA);
        MEM2REG(NBV, ldmB);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            index = (j + 1) & 1;
            WA = &AV[(index + 1) & 1];
            WB = &BV[(index + 1) & 1];
            EWA = &EAV[(index + 1) & 1];
            EWB = &EBV[(index + 1) & 1];
            MEM2REG(AV[index], COORDSMJ(ldosma, (j + 1) & 7, 7));
            MEM2REG(BV[index], COORDSMJ(ldosmb, (j + 1) & 7, 7));
            MEM2REG(EAV[index], COORDSM(ldosma, 4, (j + 1) & 7, 7));
            MEM2REG(EBV[index], COORDSM(ldosmb, 4, (j + 1) & 7, 7));
            VECSCALE(CV[0], *WA, (*WB).x);
            VECSCALE(CV[1], *EWA, (*WB).x);
            VECSCALE(CV[2], *WA, (*WB).y);
            VECSCALE(CV[3], *EWA, (*WB).y);
            VECSCALE(CV[4], *WA, (*WB).z);
            VECSCALE(CV[5], *EWA, (*WB).z);
            VECSCALE(CV[6], *WA, (*WB).w);
            VECSCALE(CV[7], *EWA, (*WB).w);
            VECSCALE(CV[8], *WA, (*EWB).x);
            VECSCALE(CV[9], *EWA, (*EWB).x);
            VECSCALE(CV[10], *WA, (*EWB).y);
            VECSCALE(CV[11], *EWA, (*EWB).y);
            VECSCALE(CV[12], *WA, (*EWB).z);
            VECSCALE(CV[13], *EWA, (*EWB).z);
            VECSCALE(CV[14], *WA, (*EWB).w);
            VECSCALE(CV[15], *EWA, (*EWB).w);
        }
        REG2MEM(ldisma, NAV);
        *COORDSMJ(ldismb, 0, 7) = NBV.x;
        *COORDSMJ(ldismb, 1, 7) = NBV.y;
        *COORDSMJ(ldismb, 2, 7) = NBV.z;
        *COORDSMJ(ldismb, 3, 7) = NBV.w;
        __syncthreads();
        MEM2REG(AV[0], COORDSMJ(ldosma, 0, 7));
        MEM2REG(BV[0], COORDSMJ(ldosmb, 0, 7));
        MEM2REG(EAV[0], COORDSM(ldosma, 4, 0, 7));
        MEM2REG(EBV[0], COORDSM(ldosmb, 4, 0, 7));
    }
    for (int j = 0; j < 8; j++) {
        index = (j + 1) & 1;
        WA = &AV[(index + 1) & 1];
        WB = &BV[(index + 1) & 1];
        EWA = &EAV[(index + 1) & 1];
        EWB = &EBV[(index + 1) & 1];
        MEM2REG(AV[index], COORDSMJ(ldosma, (j + 1) & 7, 7));
        MEM2REG(BV[index], COORDSMJ(ldosmb, (j + 1) & 7, 7));
        MEM2REG(EAV[index], COORDSM(ldosma, 4, (j + 1) & 7, 7));
        MEM2REG(EBV[index], COORDSM(ldosmb, 4, (j + 1) & 7, 7));
        VECSCALE(CV[0], *WA, (*WB).x);
        VECSCALE(CV[1], *EWA, (*WB).x);
        VECSCALE(CV[2], *WA, (*WB).y);
        VECSCALE(CV[3], *EWA, (*WB).y);
        VECSCALE(CV[4], *WA, (*WB).z);
        VECSCALE(CV[5], *EWA, (*WB).z);
        VECSCALE(CV[6], *WA, (*WB).w);
        VECSCALE(CV[7], *EWA, (*WB).w);
        VECSCALE(CV[8], *WA, (*EWB).x);
        VECSCALE(CV[9], *EWA, (*EWB).x);
        VECSCALE(CV[10], *WA, (*EWB).y);
        VECSCALE(CV[11], *EWA, (*EWB).y);
        VECSCALE(CV[12], *WA, (*EWB).z);
        VECSCALE(CV[13], *EWA, (*EWB).z);
        VECSCALE(CV[14], *WA, (*EWB).w);
        VECSCALE(CV[15], *EWA, (*EWB).w);
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
