#include "../utils.h"
#include <iostream>

void printMatrix(const float* dM, int m, int n){
    float *hM = (float*)malloc(sizeof(float)*m*n);
    cudaMemcpy(hM, dM, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < m; ++i ){
        for(int j = 0; j < n; ++j ){
            std::cout << hM[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(hM);
}
void test(const float* hA, const float* hB, int ai, int aj, int bi, int bj, int ci, int cj, int opt){
    float *A,*B,*C; 
    cudaMalloc((void**)&A, sizeof(float)*ai*aj);
    cudaMalloc((void**)&B, sizeof(float)*bi*bj);
    cudaMalloc((void**)&C, sizeof(float)*ci*cj);
    cudaMemcpy(A, hA, sizeof(float)*ai*aj, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeof(float)*bi*bj, cudaMemcpyHostToDevice);
    // matrixMul(A,B,C, bi, aj, bj, cj, opt);
    switch(opt){
        case 0:
            matrixMul(A,B,C, ai, aj, bi, bj, cj);break;
        case 1:
            matrixMulTranposeFirst(A,B,C, ai, aj, bi, bj, cj);break;
        case 2:
            matrixMulTranposeSecond(A,B,C, ai, aj, bi, bj, cj);break;
    }
    puts("----- test -----");
    printMatrix(A,ai,aj);
    printMatrix(B,bi,bj);
    printMatrix(C,ci,cj);
    puts("----- end -----");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
int main(int argc, char *argv[])
{
    float hA1[] = {
        2.,1., 1.,
        3.,1., 1.
    };
    float hB1[] = {
        1.,1.,
        1.,1.,
        2.,3.
    };
    test(
        hA1, hB1,
        2, 3, 3, 2, 2, 2,
        0
    );

    float hA2[] = {
        2.,7.,
        9.,0.,
        2.,4.
    };
    float hB2[] = {
        1.,1.,
        2.,7.,
        2.,3.
    };
    test(
        hA2, hB2,
        3, 2, 3, 2, 2, 2,
        1
    );

    return 0;
}
