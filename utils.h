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
 * Generate #`size` of random number in [0,1]
 */
void randn(float *array, int size);

/*
 * z = x*y
 *
 * yi: row number of y
 * xj: column number of x
 * yj: column number of y
 * zj: column number of z
 * transpose_opt: 1 --> transpose x
 *                2 --> transpose y
 */
void matrixMul(const float* x, const float*y, float* z, int yi, int xj, int yj, int zj, const int transpose_opt);

/* 
 * a = a + alpha*outer(x,y)
 * nrow: dimension of x
 * ncol: dimension of y
 */
void add_outer_prod(float* a, const float* x, const float* y, int nrow, int ncol, float alpha = 1.0);

/* 
 * a += c*(x - y);
 */
__global__ void add_diff(float* a, const float* x, const float* y, const float c, int size);
#endif
