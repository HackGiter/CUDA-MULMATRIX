#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <time.h>

#include "cuda_runtime.h"

#define ABS(x, y)       (x - y) > 0 ? (x - y) : (y - x)

typedef struct {
    float* devPtr;
    int width;
    int height;

} MATRIX;

void PrintMat(const MATRIX);
void InitMat(MATRIX*, int, int);
void SetMat(MATRIX*, float);
void HostRANDSetMat(MATRIX* m, unsigned long long seed);
void DestroyMat(MATRIX*);
bool VerifyMat(const MATRIX, const MATRIX);

void InitMat(MATRIX* m, int w, int h) {
    m->width = w;
    m->height = h;
    m->devPtr = (float*)malloc(sizeof(float) * (m->width * m->height));
}

void PrintMat(const MATRIX m) {
    printf("width: %d height: %d\n", m.width, m.height);
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++)
            printf("%.1f ", *(m.devPtr + i + j * m.height));
        putchar('\n');
    }
}

void SetMat(MATRIX* m, float n) {
    int size = m->width * m->height;
    for (int i = 0; i < size; i++) *(m->devPtr + i) = n;
}

void HostRANDSetMat(MATRIX* m, unsigned long long seed) {
    int size = m->width * m->height;
    srand(time(NULL));
    for (int i = 0; i < size; i++) *(m->devPtr + i) = (float)(rand() % 2) + (float)(rand() % 1000) / 1000;

}

void DestroyMat(MATRIX* m) {
    m->height = 0;
    m->width = 0;
    free(m->devPtr);
}

bool VerifyMat(const MATRIX m1, const MATRIX m2) {
    if (m1.width != m2.width || m2.height != m2.height) {
        printf("Two matrixs are different in width or height.\n");
        return false;
    }
    int size = m1.width * m2.height;
    float tmp;
    for (int i = 0; i < size; i++) {
        tmp = ABS(*(m1.devPtr + i), *(m2.devPtr + i));
        if (tmp > 1e-3) {
            printf("Two matrixs are different in values: %d %.6f %.6f %.6f\n", i, *(m1.devPtr + i), *(m2.devPtr + i), tmp);
            return false;
        }
    }
    printf("Two matrixs are exactly the same.\n");
    return true;
}
