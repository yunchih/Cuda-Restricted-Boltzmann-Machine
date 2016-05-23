#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <err.h>
#include <ctime>
#include <cmath>
#include <vector>
#include <utility>
#include <cstdint>
#include <cublas_v2.h>
#include <curand.h> 
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

/*
 * z = x*y
 *
 * xi: row number of x
 * xj: column number of x
 * yi: row number of y
 * yj: column number of y
 * zj: column number of z
 * transpose: first   --> transpose x
 *            second  --> transpose y
 *            default --> none
 */
void matrix_mul(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj);
void matrix_mul_tranpose_first(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj);
void matrix_mul_tranpose_second(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj);

/* 
 * a = a + alpha*outer(x,y)
 * nrow: dimension of x
 * ncol: dimension of y
 */
void add_outer_prod(float* a, const float* x, const float* y, int nrow, int ncol, float alpha = 1.0);


/*
 * (a-1)/b + 1
 */
__device__ __host__ int CeilDiv(int a, int b);

/*
 * Randomly initialize `array` of #size with value in [0,1]
 */
void randn(float *array, int size);

/*
 * Set up random number generator
 */
__global__ void setup_random_numbers(curandState * state, unsigned long seed);

/*
 * Transform (normalize) one example from [0,255] to [0,1]
 */
void transform_example(float* gpu_buffer, char* gpu_tmp, char* cpu_buffer, int size);
#endif
