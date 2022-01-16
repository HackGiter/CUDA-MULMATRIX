#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

__device__ inline void load_glm_to_reg(float* dst, float* src, const int n);
__device__ inline void load_glm_to_reg_transpose(float* dst, float* src, const int n, const int width);
__device__ inline void reg_xmul_reg(float*, float*, float*, const int);

__device__ inline void load_glm_to_reg(float* dst, float* src, const int n) {
#pragma unroll
    for (int i = 0; i < n; i++)
        *(dst + i) = *(src + i);
}

__device__ inline void load_glm_to_reg_transpose(float* dst, float* src, const int n, const int width) {
#pragma unroll
    for (int i = 0; i < n; i++)
        *(dst + i) = *(src + i * width);
}

__device__ inline void reg_xmul_reg(float* dst, float* src1, float* src2, const int n) {
#pragma unroll
    for (int i = 0; i < n; i++)
        *dst += *(src1 + i) * *(src2 + i);
}