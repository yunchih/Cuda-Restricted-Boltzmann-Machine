#ifndef UTILS_H
#define UTILS_H

#include <cublas_v2.h>
#include <curand.h> 


/*
 * (a-1)/b + 1
 */
__device__ __host__ int CeilDiv(int a, int b);

/*
 * Randomly initialize `array` of #size with value in [0,1]
 */
void randn(float *array, int size);

/*
 * Checking is a vector has a corrupted Infinity value
 */
bool has_nan(const float* v, size_t size);

void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line);
void curandErrCheck_(curandStatus_t stat, const char *file, int line);
#define cudaErrCheck(stat) do{ cudaErrCheck_((stat), __FILE__, __LINE__); }while(0)
#define cublasErrCheck(stat) do{ cublasErrCheck_((stat), __FILE__, __LINE__); }while(0)
#define curandErrCheck(stat) do{ curandErrCheck_((stat), __FILE__, __LINE__); }while(0)
#define KERNEL_CHECK do{ cudaErrCheck_(cudaDeviceSynchronize(), __FILE__, __LINE__);}while(0)

#endif
