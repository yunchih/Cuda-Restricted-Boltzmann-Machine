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
void randn(float *array, int size) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, array, size);
    curandDestroyGenerator(prng);
}
bool has_nan(const float* a, size_t size){
    float* res;
    cudaErrCheck(cudaMalloc((void**) &res, sizeof(float)*size));
    thrust::device_ptr<const float> p_a(a);
    thrust::device_ptr<float> p_res(res);
    thrust::transform(thrust::device, p_a, p_a+size, p_res, NaNTest());
    bool result = thrust::reduce(thrust::device, p_res, p_res+size);
    cudaErrCheck(cudaFree(res));
    return result;
}
