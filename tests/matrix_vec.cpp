#include "utils.h"
#include "blas.h"
#include <iostream>

void printMatrix(const float* dM, int m, int n){
    float *hM = (float*)malloc(sizeof(float)*m*n);
    cudaMemcpy(hM, dM, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < m; ++i ){
        for(int j = 0; j < n; ++j ){
            if(j == n-1)
                std::cout << hM[i*n + j] << ";";
            else
                std::cout << hM[i*n + j] << ",";
        }
    }
    std::cout << std::endl;
    free(hM);
}
void test(const float* hA, const float* hB, int aj, int bi, int bj, int opt, Blas& blas){
    float *A,*B,*C; 
    cudaMalloc((void**)&A, sizeof(float)*aj);
    cudaMalloc((void**)&B, sizeof(float)*bi*bj);
    cudaMalloc((void**)&C, sizeof(float)*bj);
    cudaMemcpy(A, hA, sizeof(float)*aj, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeof(float)*bi*bj, cudaMemcpyHostToDevice);
    switch(opt){
        case 0:
            blas.matrix_vec_mul(A,B,C, aj, bi, bj);break;
        case 1:
            blas.matrix_vec_mul_tranpose(A,B,C, aj, bi, bj);break;
    }
    puts("----- test -----");
    printMatrix(A,1,aj);
    printMatrix(B,bi,bj);
    printMatrix(C,1,bj);
    puts("----- end -----");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
int main(int argc, char *argv[])
{
    Blas blas;
    float hA1[] = {
        2.,1., 1., 3.,4.
    };
    float hB1[] = {
        1.,1.,0.,
        1.,1.,3.,
        2.,3.,1.,
        5.,3.,9.,
        1.,2.,3.
    };
    test(
        hA1, hB1,
        5, 5, 3,
        0,
        blas
    );
    float ha3[] = {
        2.,7.,3.,4.
    };
    float hb3[] = {
        1.,1.,2.,7.,
        2.,7.,1.,4.,
        2.,3.,2.,3.,
        1.,4.,2.,0.,
    };
    test(
        ha3, hb3,
        4, 4, 4,
        1,
        blas
    );
    float ha4[] = {
        0.6443,0.2077,0.3111,0.5949,0.0855,0.9631,0.0377,0.1068,0.0305,0.1829
    };
    float hb4[] = {
        0.6443,0.2077,0.3111,0.5949,0.0855,0.9631,0.0377,0.1068,0.0305,0.1829
        ,0.3786,0.3012,0.9234,0.2622,0.2625,0.5468,0.8852,0.6538,0.7441,0.2399
        ,0.8116,0.4709,0.4302,0.6028,0.8010,0.5211,0.9133,0.4942,0.5000,0.8865
        ,0.5328,0.2305,0.1848,0.7112,0.0292,0.2316,0.7962,0.7791,0.4799,0.0287
        ,0.3507,0.8443,0.9049,0.2217,0.9289,0.4889,0.0987,0.7150,0.9047,0.4899
        ,0.9390,0.1948,0.9797,0.1174,0.7303,0.6241,0.2619,0.9037,0.6099,0.1679
        ,0.8759,0.2259,0.4389,0.2967,0.4886,0.6791,0.3354,0.8909,0.6177,0.9787
        ,0.5502,0.1707,0.1111,0.3188,0.5785,0.3955,0.6797,0.3342,0.8594,0.7127
        ,0.6225,0.2277,0.2581,0.4242,0.2373,0.3674,0.1366,0.6987,0.8055,0.5005
        ,0.5870,0.4357,0.4087,0.5079,0.4588,0.9880,0.7212,0.1978,0.5767,0.4711
    };
    test(
        ha4, hb4,
        10, 10, 10,
        1,
        blas
    );
    return 0;
}
