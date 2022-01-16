
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <time.h>


typedef struct {
    float* devPtr;
    int size;
    int width;
    int height;

} MATRIX;

__host__ void PrintMat(const MATRIX);
__host__ void InitMat(MATRIX*, int, int);
__host__ void SetMat(MATRIX*, float);
__host__ void CUDARandInitMat(MATRIX* m, unsigned int seed);
__host__ void HostRANDInitMat(MATRIX* m, unsigned long long seed);

__host__ void InitMat(MATRIX* m, int w, int h) {
    m->width = w;
    m->height = h;
    m->size = w * h;
    m->devPtr = (float*)malloc(sizeof(float) * m->size);
}

__host__ void PrintMat(const MATRIX m) {
    printf("width: %d height: %d\n", m.width, m.height);
    for (int i = 0; i < m.height; i++) {
        int index = i * m.width;
        for (int j = 0; j < m.width; j++)
            printf("%.1f ", *(m.devPtr + index + j));
        putchar('\n');
    }
}

__host__ void SetMat(MATRIX* m, float n) {
    for (int i = 0; i < m->height; i++) {
        int index = i * m->width;
        for (int j = 0; j < m->width; j++)
            m->devPtr[index + j] = n;
    }
}

__host__ void HostRANDInitMat(MATRIX* m, unsigned long long seed) {
    //curandGenerator_t generator;
    //float* devPtr;

    //CHECK_CUDA(cudaMalloc(&devPtr, sizeof(float) * m->size));
    //CHECK_CURAND(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    //CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, seed));
    //CHECK_CURAND(curandGenerateUniform(generator, devPtr, m->size * sizeof(float)));
    //CHECK_CUDA(cudaMemcpy(m->devPtr, devPtr, sizeof(float) * m->size, cudaMemcpyDeviceToHost));
    //CHECK_CURAND(curandDestroyGenerator(generator));
    //CHECK_CUDA(cudaFree(devPtr));

    srand(time(NULL));
    for (int i = 0; i < m->size; i++) *(m->devPtr + i) = (float)(rand() % 2) + (float)(rand() % 1000) / 1000;

}


