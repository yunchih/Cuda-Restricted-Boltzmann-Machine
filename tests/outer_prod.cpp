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
void test(const float* hA, const float* hB, const float* hC, int m, int n){
    float *A,*B,*C; 
    cudaMalloc((void**)&A, sizeof(float)*m);
    cudaMalloc((void**)&B, sizeof(float)*n);
    cudaMalloc((void**)&C, sizeof(float)*m*n);
    cudaMemcpy(A, hA, sizeof(float)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(C, hC, sizeof(float)*n*m, cudaMemcpyHostToDevice);

    add_outer_prod(C,A,B,m,n,1.0);
    puts("----- test -----");
    printMatrix(A,1,m);
    printMatrix(B,1,n);
    printMatrix(C,m,n);
    puts("----- end -----");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
int main(int argc, char *argv[])
{
    float hC1[] = {
        0., 0., 1.,
        1., 1., 1.,
        0., 0., 1.
    };
    float hA1[] = {
        2.,1., 1.
    };
    float hB1[] = {
        1.,1.,1.
    };
    test(hA1, hB1, hC1, 3, 3);

    float hC2[] = {
        0., 0., 1.,2.,
        1., 1., 1.,2.,
        0., 0., 1.,2.
    };
    float hA2[] = {
        2.,1., 1.
    };
    float hB2[] = {
        1.,1.,1., 4.
    };
    test(hA2, hB2, hC2, 3, 4);
    return 0;
}
