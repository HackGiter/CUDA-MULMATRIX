#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <cublas_v2.h>
#include <crt/device_functions.h>

#include "MATRIX.cuh"
#include "util.cuh"

#define TX                        threadIdx.x
#define BX                        blockIdx.x
#define BY                        blockIdx.y

#define COORDGM(A, i, j, m)       ((A) + (j) * (m) + (i))
#define COORDGMJ(A, j, m)         ((A) + (j) * (m))
#define COORDGMI(A, i)            ((A) + (i))
#define COORDSM(S, i, j, b)       ((S) + ((j) << (b)) + (i))
#define COORDSMJ(S, j, b)         ((S) + ((j) << (b)))
#define COORDSMI(S, i)            ((S) + (i))
#define MEM2REG(R, M)              R = *((float4*)(M))
#define REG2MEM(M, R)              *((float4*)(M)) = R  
#define MEM2REG2(R, M)              R = *((float2*)(M))
#define REG2MEM2(M, R)              *((float2*)(M)) = R  
#define REG2REG(D, S)             (D) = (S);
#define VECSCALE(VO, VY, a)\
        (VO).x += a * (VY).x;\
        (VO).y += a * (VY).y;\
        (VO).z += a * (VY).z;\
        (VO).w += a * (VY).w;
#define VECSCALE2(VO, VY, a)\
        (VO).x += a * (VY).x;\
        (VO).y += a * (VY).y;
#define VCPLUSSCAL(CO, CV, alpha, beta)\
        (CO).x = alpha * (CV).x + beta * (CO).x;\
        (CO).y = alpha * (CV).y + beta * (CO).y;\
        (CO).z = alpha * (CV).z + beta * (CO).z;\
        (CO).w = alpha * (CV).w + beta * (CO).w;
#define VCPLUSSCAL2(CO, CV, alpha, beta)\
        (CO).x = alpha * (CV).x + beta * (CO).x;\
        (CO).y = alpha * (CV).y + beta * (CO).y;

#define ERROR_EXIT              -1

__global__ void warmup();
__global__ void mulMatrixKernelV7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV5_1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelV0(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelS(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelX(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelSV(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mulMatrixKernelTV(int M, int N, int K, float alpha, cudaTextureObject_t A, cudaTextureObject_t B, float beta, float* C);
__global__ void TraverseVector(cudaTextureObject_t A);

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
__global__ void TraverseVector(cudaTextureObject_t A) {
    int tx = threadIdx.x;
    float4 tmp = tex1Dfetch<float4>(A, tx);
    printf("%f %f %f %f\n", tmp.x, tmp.y, tmp.z, tmp.w);
}

__global__ void mulMatrixKernelTV(int M, int N, int K, float alpha, cudaTextureObject_t A, cudaTextureObject_t B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[8];
    memset(CV, 0, sizeof(CV));
    
    //A = COORDGM(A, (BX << 7) + ((TX & 31) << 2), TX >> 5, M);
    //B = COORDGM(B, (TX & 1) << 2, (BY << 7) + (TX >> 1), K);
    C = COORDGM(C, (BX << 7) + (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5), (BY << 7) + (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6), M);
    //int offset = M << 3;
    int coordax = (BX << 5) + (TX & 31) + (TX >> 5) * (M >> 2);
    int coordbx = (TX & 1) + ((BY << 5) + (TX >> 1)) * (K >> 2);
    int offset = M << 1;

    for (int i = 0; i < K; i += 8) {
        //CO[0] = *((float4*)A);
        CO[0] = tex1Dfetch<float4>(A, coordax);
        //CO[1] = *((float4*)B);
        CO[1] = tex1Dfetch<float4>(B, coordbx);
        *((float4*)COORDSM(sa, (TX & 31) << 2, TX >> 5, 7)) = CO[0];
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 0, 7) = CO[1].x;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 1, 7) = CO[1].y;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 2, 7) = CO[1].z;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 3, 7) = CO[1].w;
        coordax += offset;
        coordbx += 2;
        //A += offset;
        //B += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            CO[0] = *((float4*)COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), j, 7));
            CO[1] = *((float4*)COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), j, 7));
            CO[2] = *((float4*)COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, j, 7));
            CO[3] = *((float4*)COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, j, 7));
            VECSCALE(CV[0], CO[0], CO[1].x);
            VECSCALE(CV[1], CO[2], CO[1].x);
            VECSCALE(CV[2], CO[0], CO[1].y);
            VECSCALE(CV[3], CO[2], CO[1].y);
            VECSCALE(CV[4], CO[0], CO[1].z);
            VECSCALE(CV[5], CO[2], CO[1].z);
            VECSCALE(CV[6], CO[0], CO[1].w);
            VECSCALE(CV[7], CO[2], CO[1].w);
            VECSCALE(CV[8], CO[0], CO[3].x);
            VECSCALE(CV[9], CO[2], CO[3].x);
            VECSCALE(CV[10], CO[0], CO[3].y);
            VECSCALE(CV[11], CO[2], CO[3].y);
            VECSCALE(CV[12], CO[0], CO[3].z);
            VECSCALE(CV[13], CO[2], CO[3].z);
            VECSCALE(CV[14], CO[0], CO[3].w);
            VECSCALE(CV[15], CO[2], CO[3].w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGM(C, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(C, 1, M));
    CO[3] = *((float4*)COORDGM(C, 4, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(C, 1, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 1, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 2, M));
    CO[5] = *((float4*)COORDGM(C, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(C, 3, M));
    CO[7] = *((float4*)COORDGM(C, 4, 3, M));
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    *((float4*)COORDGMJ(C, 2, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(C, 3, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 3, M)) = CO[7];

    CO[0] = *((float4*)COORDGMJ(C, 4, M));
    CO[1] = *((float4*)COORDGM(C, 4, 4, M));
    CO[2] = *((float4*)COORDGMJ(C, 5, M));
    CO[3] = *((float4*)COORDGM(C, 4, 5, M));
    VCPLUSSCAL(CO[0], CV[8], alpha, beta);
    VCPLUSSCAL(CO[1], CV[9], alpha, beta);
    VCPLUSSCAL(CO[2], CV[10], alpha, beta);
    VCPLUSSCAL(CO[3], CV[11], alpha, beta);
    *((float4*)COORDGMJ(C, 4, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 4, M)) = CO[1];
    *((float4*)COORDGMJ(C, 5, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 5, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 6, M));
    CO[5] = *((float4*)COORDGM(C, 4, 6, M));
    CO[6] = *((float4*)COORDGMJ(C, 7, M));
    CO[7] = *((float4*)COORDGM(C, 4, 7, M));
    VCPLUSSCAL(CO[4], CV[12], alpha, beta);
    VCPLUSSCAL(CO[5], CV[13], alpha, beta);
    VCPLUSSCAL(CO[6], CV[14], alpha, beta);
    VCPLUSSCAL(CO[7], CV[15], alpha, beta);
    *((float4*)COORDGMJ(C, 6, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 6, M)) = CO[5];
    *((float4*)COORDGMJ(C, 7, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 7, M)) = CO[7];
}

__global__ void mulMatrixKernelSV(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[8];
    memset(CV, 0, sizeof(CV));

    A = COORDGM(A, (BX << 7) + ((TX & 31) << 2), TX >> 5, M);
    B = COORDGM(B, (TX & 1) << 2, (BY << 7) + (TX >> 1), K);
    C = COORDGM(C, (BX << 7) + (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5), (BY << 7) + (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6), M);
    int offset = M << 3;

    for (int i = 0; i < K; i += 8) {
        CO[0] = *((float4*)A);
        CO[1] = *((float4*)B);
        *((float4*)COORDSM(sa, (TX & 31) << 2, TX >> 5, 7)) = CO[0];
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 0, 7) = CO[1].x;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 1, 7) = CO[1].y;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 2, 7) = CO[1].z;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 3, 7) = CO[1].w;
        A += offset;
        B += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            CO[0] = *((float4*)COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), j, 7));
            CO[1] = *((float4*)COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), j, 7));
            CO[2] = *((float4*)COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, j, 7));
            CO[3] = *((float4*)COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, j, 7));
            VECSCALE(CV[0], CO[0], CO[1].x);
            VECSCALE(CV[1], CO[2], CO[1].x);
            VECSCALE(CV[2], CO[0], CO[1].y);
            VECSCALE(CV[3], CO[2], CO[1].y);
            VECSCALE(CV[4], CO[0], CO[1].z);
            VECSCALE(CV[5], CO[2], CO[1].z);
            VECSCALE(CV[6], CO[0], CO[1].w);
            VECSCALE(CV[7], CO[2], CO[1].w);
            VECSCALE(CV[8], CO[0], CO[3].x);
            VECSCALE(CV[9], CO[2], CO[3].x);
            VECSCALE(CV[10], CO[0], CO[3].y);
            VECSCALE(CV[11], CO[2], CO[3].y);
            VECSCALE(CV[12], CO[0], CO[3].z);
            VECSCALE(CV[13], CO[2], CO[3].z);
            VECSCALE(CV[14], CO[0], CO[3].w);
            VECSCALE(CV[15], CO[2], CO[3].w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGM(C, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(C, 1, M));
    CO[3] = *((float4*)COORDGM(C, 4, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(C, 1, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 1, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 2, M));
    CO[5] = *((float4*)COORDGM(C, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(C, 3, M));
    CO[7] = *((float4*)COORDGM(C, 4, 3, M));
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    *((float4*)COORDGMJ(C, 2, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(C, 3, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 3, M)) = CO[7];

    CO[0] = *((float4*)COORDGMJ(C, 4, M));
    CO[1] = *((float4*)COORDGM(C, 4, 4, M));
    CO[2] = *((float4*)COORDGMJ(C, 5, M));
    CO[3] = *((float4*)COORDGM(C, 4, 5, M));
    VCPLUSSCAL(CO[0], CV[8], alpha, beta);
    VCPLUSSCAL(CO[1], CV[9], alpha, beta);
    VCPLUSSCAL(CO[2], CV[10], alpha, beta);
    VCPLUSSCAL(CO[3], CV[11], alpha, beta);
    *((float4*)COORDGMJ(C, 4, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 4, M)) = CO[1];
    *((float4*)COORDGMJ(C, 5, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 5, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 6, M));
    CO[5] = *((float4*)COORDGM(C, 4, 6, M));
    CO[6] = *((float4*)COORDGMJ(C, 7, M));
    CO[7] = *((float4*)COORDGM(C, 4, 7, M));
    VCPLUSSCAL(CO[4], CV[12], alpha, beta);
    VCPLUSSCAL(CO[5], CV[13], alpha, beta);
    VCPLUSSCAL(CO[6], CV[14], alpha, beta);
    VCPLUSSCAL(CO[7], CV[15], alpha, beta);
    *((float4*)COORDGMJ(C, 6, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 6, M)) = CO[5];
    *((float4*)COORDGMJ(C, 7, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 7, M)) = CO[7];
}

__global__ void mulMatrixKernelX(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[4], CO, AV, BV;
    memset(CV, 0, sizeof(CV));

    A = COORDGM(A, (BX << 6) + ((TX & 15) << 2), TX >> 4, M);
    B = COORDGM(B, (TX & 3) << 2, (TX >> 2) + (BY << 6), K);
    C = COORDGM(C, (BX << 6) + ((TX & 15) << 2), ((TX >> 4) << 2) + (BY << 6), M);
    int offset = M << 4;

    for (int i = 0; i < K; i += 16) {
        MEM2REG(AV, A);
        MEM2REG(BV, B);
        *((float4*)COORDSM(sa, (TX & 15) << 2, TX >> 4, 6)) = AV;
        *COORDSMJ(COORDSM(sb, TX >> 2, (TX & 3) << 2, 6), 0, 6) = BV.x;
        *COORDSMJ(COORDSM(sb, TX >> 2, (TX & 3) << 2, 6), 1, 6) = BV.y;
        *COORDSMJ(COORDSM(sb, TX >> 2, (TX & 3) << 2, 6), 2, 6) = BV.z;
        *COORDSMJ(COORDSM(sb, TX >> 2, (TX & 3) << 2, 6), 3, 6) = BV.w;
        A += offset;
        B += 16;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j++) {
            MEM2REG(AV, COORDSMJ(COORDSMI(sa, (TX & 15) << 2), j, 6));
            MEM2REG(BV, COORDSMJ(COORDSMI(sb, ((TX >> 4) << 2)), j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
        }
        __syncthreads();
    }
    MEM2REG(CO, COORDGMJ(C, 0, M));
    MEM2REG(AV, COORDGMJ(C, 1, M));
    VCPLUSSCAL(CO, CV[0], alpha, beta);
    VCPLUSSCAL(AV, CV[1], alpha, beta);
    REG2MEM(COORDGMJ(C, 0, M), CO);
    MEM2REG(BV, COORDGMJ(C, 2, M));
    REG2MEM(COORDGMJ(C, 1, M), AV);
    MEM2REG(CO, COORDGMJ(C, 3, M));
    VCPLUSSCAL(BV, CV[2], alpha, beta);
    VCPLUSSCAL(CO, CV[3], alpha, beta);
    REG2MEM(COORDGMJ(C, 2, M), BV);
    REG2MEM(COORDGMJ(C, 3, M), CO);
}

__global__ void mulMatrixKernelS(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float CV = 0, AV, BV;
    int tx = threadIdx.x, bx = blockIdx.x << 5, by = blockIdx.y << 5, offset = M << 5;
    int coordax = tx & 31, coorday = tx >> 5,
        coordbx = tx >> 5, coordby = tx & 31;
    A = A + coordax + bx + coorday * M;
    B = B + coordbx + (coordby + by) * K;
    C = C + coorday + bx + (coordax + by) * M;
    float* ldisma = sa + coordax + (coorday << 5),
         * ldismb = sb + coordby + (coordbx << 5),
         * ldosma = sa + coorday,
         * ldosmb = sb + coordax;

    for (int i = 0; i < K; i += 32) {
        AV = *A;
        BV = *B;
        *ldisma = AV;
        *ldismb = BV;
        A += offset;
        B += 32;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 32; j++) {
            AV = *COORDSMJ(ldosma, j, 5);
            BV = *COORDSMJ(ldosmb, j, 5);
            CV += (double)AV * BV;
        }
        __syncthreads();
    }
    AV = *C;
    AV = alpha * CV + beta * AV;
    *C = AV;
}

__global__ void mulMatrixKernelV0(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[512];
    __shared__ float sb[512];
    float2 GAV, GBV;
    float4 CV[4], CO[4], AV, BV;
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int coordax = (tx & 31) << 1, coorday = tx >> 5, 
        coordbx = (tx & 3) << 1, coordby = tx >> 2,
        coordcx = (tx & 15) << 2, coordcy = (tx >> 4) << 2;

    float* ldisma = COORDSM(sa, coordax, coorday, 6),
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordcx),
         * ldosmb = COORDSMI(sb, coordcy);
    float* ldmA = COORDGM(A, coordax + bx, coorday, M),
         * ldmB = COORDGM(B, coordbx, coordby + by, K),
         * ldmC = COORDGM(C, bx + coordcx, coordcy + by, M);
    int offset = M << 3;

    for (int i = 0; i < K; i += 8) {
        MEM2REG2(GAV, ldmA);
        MEM2REG2(GBV, ldmB);
        REG2MEM2(ldisma, GAV);
        *COORDSMJ(ldismb, 0, 6) = GBV.x;
        *COORDSMJ(ldismb, 1, 6) = GBV.y;
        ldmA += offset;
        ldmB += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MEM2REG(AV, COORDSMJ(ldosma, j, 6));
            MEM2REG(BV, COORDSMJ(ldosmb, j, 6));
            VECSCALE(CV[0], AV, BV.x);
            VECSCALE(CV[1], AV, BV.y);
            VECSCALE(CV[2], AV, BV.z);
            VECSCALE(CV[3], AV, BV.w);
        }
        __syncthreads();
    }
    MEM2REG(CO[0], COORDGMJ(ldmC, 0, M));
    MEM2REG(CO[1], COORDGMJ(ldmC, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    MEM2REG(CO[2], COORDGMJ(ldmC, 2, M));
    MEM2REG(CO[3], COORDGMJ(ldmC, 3, M));
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    REG2MEM(COORDGMJ(ldmC, 0, M), CO[0]);
    REG2MEM(COORDGMJ(ldmC, 1, M), CO[1]);
    REG2MEM(COORDGMJ(ldmC, 2, M), CO[2]);
    REG2MEM(COORDGMJ(ldmC, 3, M), CO[3]);
}

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
    float4 CV[4], CO[4], AV, EAV, BV, EBV, TBV;
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

__global__ void mulMatrixKernelV5_1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[4], CO[4];
    memset(CV, 0, sizeof(CV));
    int tx = threadIdx.x, bx = blockIdx.x << 6, by = blockIdx.y << 6;
    int coordax = (tx & 15) << 2, coorday = tx >> 4, coordbx = (tx & 3) << 2, coordby = tx >> 2;

    float* ldisma = COORDSM(sa, coordax, coorday, 6),
         * ldismb = COORDSM(sb, coordby, coordbx, 6),
         * ldosma = COORDSMI(sa, coordax),
         * ldosmb = COORDSMI(sb, (coorday << 2));
    A = COORDGM(A, (bx + coordax), coorday, M),
    B = COORDGM(B, coordbx, (coordby + by), K),
    C = COORDGM(C, (bx + coordax), ((coorday << 2) + by), M);
    int offset = M << 4;

    for (int i = 0; i < K; i += 16) {
        MEM2REG(CO[0], A);
        MEM2REG(CO[1], B);
        *((float4*)ldisma) = CO[0];
        *COORDSMJ(ldismb, 0, 6) = CO[1].x;
        *COORDSMJ(ldismb, 1, 6) = CO[1].y;
        *COORDSMJ(ldismb, 2, 6) = CO[1].z;
        *COORDSMJ(ldismb, 3, 6) = CO[1].w;
        A += offset;
        B += 16;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j++) {
            CO[0] = *((float4*)COORDSMJ(ldosma, j, 6));
            CO[1] = *((float4*)COORDSMJ(ldosmb, j, 6));
            VECSCALE(CV[0], CO[0], CO[1].x);
            VECSCALE(CV[1], CO[0], CO[1].y);
            VECSCALE(CV[2], CO[0], CO[1].z);
            VECSCALE(CV[3], CO[0], CO[1].w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGMJ(C, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    CO[2] = *((float4*)COORDGMJ(C, 2, M));
    CO[3] = *((float4*)COORDGMJ(C, 3, M));
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGMJ(C, 1, M)) = CO[1];
    *((float4*)COORDGMJ(C, 2, M)) = CO[2];
    *((float4*)COORDGMJ(C, 3, M)) = CO[3];
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
    A = COORDGM(A, (bx + coordax), coorday, M),
        B = COORDGM(B, coordbx, (coordby + by), K),
        C = COORDGM(C, (bx + coordax), ((coorday << 2) + by), M);
    int offset = M << 4;

    for (int i = 0; i < K; i += 16) {
        MEM2REG(AV, A);
        MEM2REG(BV, B);
        *((float4*)ldisma) = AV;
        *COORDSMJ(ldismb, 0, 6) = BV.x;
        *COORDSMJ(ldismb, 1, 6) = BV.y;
        *COORDSMJ(ldismb, 2, 6) = BV.z;
        *COORDSMJ(ldismb, 3, 6) = BV.w;
        A += offset;
        B += 16;
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
    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGMJ(C, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    CO[2] = *((float4*)COORDGMJ(C, 2, M));
    CO[3] = *((float4*)COORDGMJ(C, 3, M));
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGMJ(C, 1, M)) = CO[1];
    *((float4*)COORDGMJ(C, 2, M)) = CO[2];
    *((float4*)COORDGMJ(C, 3, M)) = CO[3];
}

__global__ void mulMatrixKernelV6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[8];
    memset(CV, 0, sizeof(CV));

    A = COORDGM(A, (BX << 7) + ((TX & 31) << 2), TX >> 5, M);
    B = COORDGM(B, (TX & 1) << 2, (BY << 7) + (TX >> 1), K);
    C = COORDGM(C, (BX << 7) + (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5), (BY << 7) + (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6), M);
    int offset = M << 3;

    for (int i = 0; i < K; i += 8) {
        CO[0] = *((float4*)A);
        CO[1] = *((float4*)B);
        *((float4*)COORDSM(sa, (TX & 31) << 2, TX >> 5, 7)) = CO[0];
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 0, 7) = CO[1].x;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 1, 7) = CO[1].y;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 2, 7) = CO[1].z;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 3, 7) = CO[1].w;
        A += offset;
        B += 8;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            CO[0] = *((float4*)COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), j, 7));
            CO[1] = *((float4*)COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), j, 7));
            CO[2] = *((float4*)COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, j, 7));
            CO[3] = *((float4*)COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, j, 7));
            VECSCALE(CV[0], CO[0], CO[1].x);
            VECSCALE(CV[1], CO[2], CO[1].x);
            VECSCALE(CV[2], CO[0], CO[1].y);
            VECSCALE(CV[3], CO[2], CO[1].y);
            VECSCALE(CV[4], CO[0], CO[1].z);
            VECSCALE(CV[5], CO[2], CO[1].z);
            VECSCALE(CV[6], CO[0], CO[1].w);
            VECSCALE(CV[7], CO[2], CO[1].w);
            VECSCALE(CV[8], CO[0], CO[3].x);
            VECSCALE(CV[9], CO[2], CO[3].x);
            VECSCALE(CV[10], CO[0], CO[3].y);
            VECSCALE(CV[11], CO[2], CO[3].y);
            VECSCALE(CV[12], CO[0], CO[3].z);
            VECSCALE(CV[13], CO[2], CO[3].z);
            VECSCALE(CV[14], CO[0], CO[3].w);
            VECSCALE(CV[15], CO[2], CO[3].w);
        }
        __syncthreads();
    }
    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGM(C, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(C, 1, M));
    CO[3] = *((float4*)COORDGM(C, 4, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(C, 1, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 1, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 2, M));
    CO[5] = *((float4*)COORDGM(C, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(C, 3, M));
    CO[7] = *((float4*)COORDGM(C, 4, 3, M));
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    *((float4*)COORDGMJ(C, 2, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(C, 3, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 3, M)) = CO[7];

    CO[0] = *((float4*)COORDGMJ(C, 4, M));
    CO[1] = *((float4*)COORDGM(C, 4, 4, M));
    CO[2] = *((float4*)COORDGMJ(C, 5, M));
    CO[3] = *((float4*)COORDGM(C, 4, 5, M));
    VCPLUSSCAL(CO[0], CV[8], alpha, beta);
    VCPLUSSCAL(CO[1], CV[9], alpha, beta);
    VCPLUSSCAL(CO[2], CV[10], alpha, beta);
    VCPLUSSCAL(CO[3], CV[11], alpha, beta);
    *((float4*)COORDGMJ(C, 4, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 4, M)) = CO[1];
    *((float4*)COORDGMJ(C, 5, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 5, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 6, M));
    CO[5] = *((float4*)COORDGM(C, 4, 6, M));
    CO[6] = *((float4*)COORDGMJ(C, 7, M));
    CO[7] = *((float4*)COORDGM(C, 4, 7, M));
    VCPLUSSCAL(CO[4], CV[12], alpha, beta);
    VCPLUSSCAL(CO[5], CV[13], alpha, beta);
    VCPLUSSCAL(CO[6], CV[14], alpha, beta);
    VCPLUSSCAL(CO[7], CV[15], alpha, beta);
    *((float4*)COORDGMJ(C, 6, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 6, M)) = CO[5];
    *((float4*)COORDGMJ(C, 7, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 7, M)) = CO[7];
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
    A = COORDGM(A, bx + coordax, coorday, M);
    B = COORDGM(B, coordbx, by + coordby, K);
    C = COORDGM(C, bx + coordcx, by + coordcy, M);
    float4* WA, * WB, * EWA, * EWB;
    int offset = M << 3, index;

    MEM2REG(NAV, A);
    MEM2REG(NBV, B);
    REG2MEM(ldisma, NAV);
    *COORDSMJ(ldismb, 0, 7) = NBV.x;
    *COORDSMJ(ldismb, 1, 7) = NBV.y;
    *COORDSMJ(ldismb, 2, 7) = NBV.z;
    *COORDSMJ(ldismb, 3, 7) = NBV.w;
    for (int i = 8; i < K; i += 8) {
        A += offset;
        B += 8;
        __syncthreads();
        MEM2REG(NAV, A);
        MEM2REG(NBV, B);
        MEM2REG(AV[0], COORDSMJ(ldosma, 0, 7));
        MEM2REG(BV[0], COORDSMJ(ldosmb, 0, 7));
        MEM2REG(EAV[0], COORDSM(ldosma, 4, 0, 7));
        MEM2REG(EBV[0], COORDSM(ldosmb, 4, 0, 7));
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
        __syncthreads();
        REG2MEM(ldisma, NAV);
        *COORDSMJ(ldismb, 0, 7) = NBV.x;
        *COORDSMJ(ldismb, 1, 7) = NBV.y;
        *COORDSMJ(ldismb, 2, 7) = NBV.z;
        *COORDSMJ(ldismb, 3, 7) = NBV.w;
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

    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGM(C, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(C, 1, M));
    CO[3] = *((float4*)COORDGM(C, 4, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(C, 1, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 1, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 2, M));
    CO[5] = *((float4*)COORDGM(C, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(C, 3, M));
    CO[7] = *((float4*)COORDGM(C, 4, 3, M));
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    *((float4*)COORDGMJ(C, 2, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(C, 3, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 3, M)) = CO[7];

    CO[8] = *((float4*)COORDGMJ(C, 4, M));
    CO[9] = *((float4*)COORDGM(C, 4, 4, M));
    CO[10] = *((float4*)COORDGMJ(C, 5, M));
    CO[11] = *((float4*)COORDGM(C, 4, 5, M));
    VCPLUSSCAL(CO[8], CV[8], alpha, beta);
    VCPLUSSCAL(CO[9], CV[9], alpha, beta);
    VCPLUSSCAL(CO[10], CV[10], alpha, beta);
    VCPLUSSCAL(CO[11], CV[11], alpha, beta);
    *((float4*)COORDGMJ(C, 4, M)) = CO[8];
    *((float4*)COORDGM(C, 4, 4, M)) = CO[9];
    *((float4*)COORDGMJ(C, 5, M)) = CO[10];
    *((float4*)COORDGM(C, 4, 5, M)) = CO[11];

    CO[12] = *((float4*)COORDGMJ(C, 6, M));
    CO[13] = *((float4*)COORDGM(C, 4, 6, M));
    CO[14] = *((float4*)COORDGMJ(C, 7, M));
    CO[15] = *((float4*)COORDGM(C, 4, 7, M));
    VCPLUSSCAL(CO[12], CV[12], alpha, beta);
    VCPLUSSCAL(CO[13], CV[13], alpha, beta);
    VCPLUSSCAL(CO[14], CV[14], alpha, beta);
    VCPLUSSCAL(CO[15], CV[15], alpha, beta);
    *((float4*)COORDGMJ(C, 6, M)) = CO[12];
    *((float4*)COORDGM(C, 4, 6, M)) = CO[13];
    *((float4*)COORDGMJ(C, 7, M)) = CO[14];
    *((float4*)COORDGM(C, 4, 7, M)) = CO[15];
}

__global__ void mulMatrixKernelXI(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    float4 CV[16], CO[12];
    memset(CV, 0, sizeof(CV));

    A = COORDGM(A, (BX << 7) + ((TX & 31) << 2), TX >> 5, M);
    B = COORDGM(B, (TX & 1) << 2, (BY << 7) + (TX >> 1), K);
    C = COORDGM(C, (BX << 7) + (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5), (BY << 7) + (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6), M);
    int offset = M << 3;

    MEM2REG(CO[8], A);
    MEM2REG(CO[9], B);
    REG2MEM(COORDSM(sa, (TX & 31) << 2, TX >> 5, 7), CO[8]);
    *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 0, 7) = CO[9].x;
    *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 1, 7) = CO[9].y;
    *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 2, 7) = CO[9].z;
    *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 3, 7) = CO[9].w;
    for (int i = 8; i < K; i += 8) {
        A += offset;
        B += 8;
        __syncthreads();
        MEM2REG(CO[8], A);
        MEM2REG(CO[9], B);
        MEM2REG(CO[0], COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 0, 7));
        MEM2REG(CO[2], COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 0, 7));
        MEM2REG(CO[4], COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, 0, 7));
        MEM2REG(CO[6], COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, 0, 7));
#pragma unroll
        for (int j = 1; j < 8; j++) {
            MEM2REG(CO[0 + (j & 1)], COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), j, 7));
            MEM2REG(CO[2 + (j & 1)], COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), j, 7));
            MEM2REG(CO[4 + (j & 1)], COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, j, 7));
            MEM2REG(CO[6 + (j & 1)], COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, j, 7));
            VECSCALE(CV[0], CO[0 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].x);
            VECSCALE(CV[1], CO[4 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].x);
            VECSCALE(CV[2], CO[0 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].y);
            VECSCALE(CV[3], CO[4 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].y);
            VECSCALE(CV[4], CO[0 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].z);
            VECSCALE(CV[5], CO[4 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].z);
            VECSCALE(CV[6], CO[0 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].w);
            VECSCALE(CV[7], CO[4 + ((j + 1) & 1)], CO[2 + ((j + 1) & 1)].w);
            VECSCALE(CV[8], CO[0 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].x);
            VECSCALE(CV[9], CO[4 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].x);
            VECSCALE(CV[10], CO[0 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].y);
            VECSCALE(CV[11], CO[4 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].y);
            VECSCALE(CV[12], CO[0 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].z);
            VECSCALE(CV[13], CO[4 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].z);
            VECSCALE(CV[14], CO[0 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].w);
            VECSCALE(CV[15], CO[4 + ((j + 1) & 1)], CO[6 + ((j + 1) & 1)].w);
        }
        VECSCALE(CV[0], CO[1], CO[3].x);
        VECSCALE(CV[1], CO[5], CO[3].x);
        VECSCALE(CV[2], CO[1], CO[3].y);
        VECSCALE(CV[3], CO[5], CO[3].y);
        VECSCALE(CV[4], CO[1], CO[3].z);
        VECSCALE(CV[5], CO[5], CO[3].z);
        VECSCALE(CV[6], CO[1], CO[3].w);
        VECSCALE(CV[7], CO[5], CO[3].w);
        VECSCALE(CV[8], CO[1], CO[7].x);
        VECSCALE(CV[9], CO[5], CO[7].x);
        VECSCALE(CV[10], CO[1], CO[7].y);
        VECSCALE(CV[11], CO[5], CO[7].y);
        VECSCALE(CV[12], CO[1], CO[7].z);
        VECSCALE(CV[13], CO[5], CO[7].z);
        VECSCALE(CV[14], CO[1], CO[7].w);
        VECSCALE(CV[15], CO[5], CO[7].w);

        __syncthreads();
        REG2MEM(COORDSM(sa, (TX & 31) << 2, TX >> 5, 7), CO[8]);
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 0, 7) = CO[9].x;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 1, 7) = CO[9].y;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 2, 7) = CO[9].z;
        *COORDSMJ(COORDSM(sb, TX >> 1, (TX & 1) << 2, 7), 3, 7) = CO[9].w;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
        CO[0] = *((float4*)COORDSMJ(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), j, 7));
        CO[1] = *((float4*)COORDSMJ(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), j, 7));
        CO[2] = *((float4*)COORDSM(COORDSMI(sa, (((TX & 31) & 3) << 3) + (((TX >> 5) & 3) << 5)), 4, j, 7));
        CO[3] = *((float4*)COORDSM(COORDSMI(sb, (((TX & 31) / 4) << 3) + (((TX >> 5) / 4) << 6)), 4, j, 7));
        VECSCALE(CV[0], CO[0], CO[1].x);
        VECSCALE(CV[1], CO[2], CO[1].x);
        VECSCALE(CV[2], CO[0], CO[1].y);
        VECSCALE(CV[3], CO[2], CO[1].y);
        VECSCALE(CV[4], CO[0], CO[1].z);
        VECSCALE(CV[5], CO[2], CO[1].z);
        VECSCALE(CV[6], CO[0], CO[1].w);
        VECSCALE(CV[7], CO[2], CO[1].w);
        VECSCALE(CV[8], CO[0], CO[3].x);
        VECSCALE(CV[9], CO[2], CO[3].x);
        VECSCALE(CV[10], CO[0], CO[3].y);
        VECSCALE(CV[11], CO[2], CO[3].y);
        VECSCALE(CV[12], CO[0], CO[3].z);
        VECSCALE(CV[13], CO[2], CO[3].z);
        VECSCALE(CV[14], CO[0], CO[3].w);
        VECSCALE(CV[15], CO[2], CO[3].w);
    }

    CO[0] = *((float4*)COORDGMJ(C, 0, M));
    CO[1] = *((float4*)COORDGM(C, 4, 0, M));
    CO[2] = *((float4*)COORDGMJ(C, 1, M));
    CO[3] = *((float4*)COORDGM(C, 4, 1, M));
    VCPLUSSCAL(CO[0], CV[0], alpha, beta);
    VCPLUSSCAL(CO[1], CV[1], alpha, beta);
    VCPLUSSCAL(CO[2], CV[2], alpha, beta);
    VCPLUSSCAL(CO[3], CV[3], alpha, beta);
    *((float4*)COORDGMJ(C, 0, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 0, M)) = CO[1];
    *((float4*)COORDGMJ(C, 1, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 1, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 2, M));
    CO[5] = *((float4*)COORDGM(C, 4, 2, M));
    CO[6] = *((float4*)COORDGMJ(C, 3, M));
    CO[7] = *((float4*)COORDGM(C, 4, 3, M));
    VCPLUSSCAL(CO[4], CV[4], alpha, beta);
    VCPLUSSCAL(CO[5], CV[5], alpha, beta);
    VCPLUSSCAL(CO[6], CV[6], alpha, beta);
    VCPLUSSCAL(CO[7], CV[7], alpha, beta);
    *((float4*)COORDGMJ(C, 2, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 2, M)) = CO[5];
    *((float4*)COORDGMJ(C, 3, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 3, M)) = CO[7];

    CO[0] = *((float4*)COORDGMJ(C, 4, M));
    CO[1] = *((float4*)COORDGM(C, 4, 4, M));
    CO[2] = *((float4*)COORDGMJ(C, 5, M));
    CO[3] = *((float4*)COORDGM(C, 4, 5, M));
    VCPLUSSCAL(CO[0], CV[8], alpha, beta);
    VCPLUSSCAL(CO[1], CV[9], alpha, beta);
    VCPLUSSCAL(CO[2], CV[10], alpha, beta);
    VCPLUSSCAL(CO[3], CV[11], alpha, beta);
    *((float4*)COORDGMJ(C, 4, M)) = CO[0];
    *((float4*)COORDGM(C, 4, 4, M)) = CO[1];
    *((float4*)COORDGMJ(C, 5, M)) = CO[2];
    *((float4*)COORDGM(C, 4, 5, M)) = CO[3];

    CO[4] = *((float4*)COORDGMJ(C, 6, M));
    CO[5] = *((float4*)COORDGM(C, 4, 6, M));
    CO[6] = *((float4*)COORDGMJ(C, 7, M));
    CO[7] = *((float4*)COORDGM(C, 4, 7, M));
    VCPLUSSCAL(CO[4], CV[12], alpha, beta);
    VCPLUSSCAL(CO[5], CV[13], alpha, beta);
    VCPLUSSCAL(CO[6], CV[14], alpha, beta);
    VCPLUSSCAL(CO[7], CV[15], alpha, beta);
    *((float4*)COORDGMJ(C, 6, M)) = CO[4];
    *((float4*)COORDGM(C, 4, 6, M)) = CO[5];
    *((float4*)COORDGMJ(C, 7, M)) = CO[6];
    *((float4*)COORDGM(C, 4, 7, M)) = CO[7];
}