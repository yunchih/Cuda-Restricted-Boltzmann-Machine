#include "utils.h"
#include "debug.h"
#include "messages.h"

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
static const char* cublasGetErrorEnum(cublasStatus_t error){
    switch (error){
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      throw_error("CUDA Error: " << cudaGetErrorString(stat) << " at " << file << ":" << line);
      exit(1);
   }
}
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      throw_error("cuBlas Error: " << cublasGetErrorEnum(stat) << " at " << file << ":" << line);
      exit(1);
   }
}
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      throw_error("cuRand Error: " << stat << " at " << file << ":" << line);
      exit(1);
   }
}

cublasHandle_t& cublas_handle(){
    static cublasHandle_t handle = NULL;
    if(handle == NULL)
        cublasErrCheck(cublasCreate(&handle));
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
    if(stat != CUBLAS_STATUS_SUCCESS){
        throw_error("CUBLAS outer prodduct error");
        exit(1);
    }
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
    curand_init( seed, id, 0, &state[id] );
} 

__global__ void transform_example_kernel(float* to, char* from, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size ){
        to[i] = ((float)(from[i])) * 2.0f / 255.0f - 1.0f;
    }
}

void transform_example(float* gpu_buffer, char* gpu_tmp, char* cpu_buffer, int size){
    const int bsize = 128;
    const int gsize = CeilDiv(size,bsize);
    cudaErrCheck(cudaMemcpy(gpu_tmp, cpu_buffer, size, cudaMemcpyHostToDevice));
    transform_example_kernel<<<bsize, gsize>>>(gpu_buffer, gpu_tmp, size);
}
