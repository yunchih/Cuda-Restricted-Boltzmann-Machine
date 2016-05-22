#include "utils.h"

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s at %s %d\n", cudaGetErrorString(stat), file, line);
      exit(1);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d at %s %d\n", stat, file, line);
      exit(1);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d at %s %d\n", stat, file, line);
      exit(1);
   }
}

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }

cublasHandle_t& cublas_handle(){
    static cublasHandle_t handle = NULL;
    if(handle == NULL){
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if(stat != CUBLAS_STATUS_SUCCESS)
            errx(1, "CUBLAS initialization failed\n");
    }
    return handle;
}
// z(m,n) = x(m,k) * y(k,n)
void matrix_mul(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yj, n = xi, k = yi;
    cublasErrCheck(cublasSgemm(
        cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        y, yj,
        x, xj,
        &beta,
        z, zj
    ));
}
void matrix_mul_tranpose_first(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yj, n = xj, k = yi;
    cublasErrCheck(cublasSgemm(
        cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        m, n, k,
        &alpha,
        y, yj,
        x, xj,
        &beta,
        z, zj
    ));
}
void matrix_mul_tranpose_second(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yi, n = xi, k = yj;
    cublasErrCheck(cublasSgemm(
        cublas_handle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        y, yj,
        x, xj,
        &beta,
        z, zj
    ));
}
// a = a + outer(x,y)
void add_outer_prod(float* a, const float* x, const float* y, int nrow, int ncol, float alpha){
    cublasStatus_t stat;
    stat = cublasSger(
        cublas_handle(),
        ncol, nrow,
        &alpha,
        y, 1,
        x, 1,
        a, ncol
    );
    if(stat != CUBLAS_STATUS_SUCCESS)
        errx(1,"CUBLAS outer prodduct error\n");
}

void randn(float *array, int size) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, array, size);
    curandDestroyGenerator(prng);
}
__global__ void setup_random_numbers(curandState * state, unsigned long seed){
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

