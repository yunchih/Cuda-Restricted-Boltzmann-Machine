#ifndef BLAS_H
#define BLAS_H

#include <cublas_v2.h>
#include <curand.h> 

struct Blas {
    
    cublasHandle_t handle;

    Blas(){
        cublasErrCheck(cublasCreate(&handle));
    }
    ~Blas(){
        cublasErrCheck(cublasDestroy(handle));
    }
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
    // z(m,n) = x(m,k) * y(k,n)
    void matrix_mul(const float* x, const float*y, float* z, int xi, int xj, int yi, int yj, int zj){
        float alpha = 1.0, beta = 0.0;
        int m = yj, n = xi, k = yi;
        cublasErrCheck(cublasSgemm(
            this->handle,
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
            this->handle,
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
            this->handle,
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
    /* 
     * a = a + alpha*outer(x,y)
     * nrow: dimension of x
     * ncol: dimension of y
     */
    void add_outer_prod(float* a, const float* x, const float* y, int nrow, int ncol, float alpha){
        cublasErrCheck(cublasSger(
            this->handle,
            ncol, nrow,
            &alpha,
            y, 1,
            x, 1,
            a, ncol
        ));
    }
};
#endif
