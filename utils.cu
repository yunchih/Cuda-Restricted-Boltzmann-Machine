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

__forceinline__ __device__ float sigmoidf(float in) {
    return in / (1.f + fabsf(in));  
}
__global__ void vectorAdd(float *y, float *a,  float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void sigmoid(float *y, float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoidf(a[i]);
}
__global__
struct do_sample{
    __host__ __device__
    int operator()(const float n) const{
        return n > 0.5;  
    }
};
__global__
struct sigmoid{
    __host__ __device__
    float operator()(const float n) const{
        return n / (1.f + abs(n));  
    }
};
cublasHandle_t& cublasHandle(){
    static cublasHandle_t handle = NULL;
    if(handle == NULL){
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if(stat != CUBLAS_STATUS_SUCCESS)
            errx(1, "CUBLAS initialization failed\n");
    }
    return handle;
}
void randn(float *array, int size) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, array, size);
    curandDestroyGenerator(prng);
}
// z(m,n) = x(m,k) * y(k,n)
void matrixMul(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yj, n = xi, k = yi;
    cublasErrCheck(cublasSgemm(
        cublasHandle(),
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
void matrixMulTranposeFirst(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yj, n = xj, k = yi;
    cublasErrCheck(cublasSgemm(
        cublasHandle(),
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
void matrixMulTranposeSecond(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
    float alpha = 1.0, beta = 0.0;
    int m = yi, n = xi, k = yj;
    cublasErrCheck(cublasSgemm(
        cublasHandle(),
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
// x = sigmoid(x + y)
__global__ void add_sigmoid(float* x, const float* y, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size )
        x[i] += sigmoidf(x[i] + y[i]);
}
__global__ void add_diff(float* a, const float* x, const float* y, const float c, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size )
        a[i] += c*(x[i] - y[i]);
}
// a = a + outer(x,y)
void add_outer_prod(float* a, const float* x, const float* y, int nrow, int ncol, float alpha){
    cublasStatus_t stat;
    stat = cublasSger(
        cublasHandle(),
        ncol, nrow,
        &alpha,
        y, 1,
        x, 1,
        a, ncol
    );
    if(stat != CUBLAS_STATUS_SUCCESS)
        errx(1,"CUBLAS outer prodduct error\n");
}
