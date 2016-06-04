#ifndef RBM_H
#define RBM_H

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cassert>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <iomanip>
#include <utility>
#include <cmath>
#include <ctime>
#include "utils.h"
#include "blas.h"
#include "mnist_reader.h"

class RBM {

public:
    RBM(int _n_visible, int _n_hidden, float _learning_rate, 
        int _n_epoch, int _n_CD, int _sample_size, 
        MnistReader& _train_reader, MnistReader& _test_reader, 
        std::pair<int,int> out_img_dimension);
    ~RBM();
    void train();

private:
    void update_w(const float* h_0, const float* v_0, const float* h_k, const float* v_k);
    void update_b(const float* v_0, const float* v_k);
    void update_c(const float* h_0, const float* h_k);
    void do_contrastive_divergence(const float* v_0);
    float* reconstruct(const float* v_0);
    void write_reconstruct_image(int epoch, float cost);
    void sample_h(float* h_s, const float* h_0);
    float calculate_cost_each(const float* v_0);
    float calculate_cost();
    void train_step();

    template <bool do_sample>
    float* get_h_given_v(float* h, const float* v);

    template <bool do_sample>
    float* get_v_given_h(const float* h, float* v);


/* Fields */
    int n_visible, n_hidden;
    int n_epoch, n_train_data, n_CD, n_sample;
    float learning_rate;

    /* pW: _n_visible * _n_hidden matrix */
    /* pb: _n_visible vector */
    /* pc: _n_hidden vector  */
    float *pW, *pB, *pC;

    std::pair<int,int> out_img_d;
    MnistReader& test_reader;
    MnistReader& train_reader;
    Blas blas;
};

/*
 *   ============================
 *   Helper functions for rbm.cu
 *   ============================
 */
template <bool do_sample>
static __global__ void add_sigmoid(float* x, const float* y, int size);
static __global__ void add_diff(float* a, const float* x, const float* y, const float c, int size);
static __global__ void vec_sample(float* v, int size);
static __global__ void random_fill_range(float* w, int size, float low, float high);
__forceinline__ __device__ float get_sample(float f);
__forceinline__ __device__ float get_rand();
__forceinline__ __device__ float sigmoidf(float in);

/*
 * Helper struct used to compute square error
 */
struct Square_diff{
    __host__ __device__ float operator()(const float &lhs, const float &rhs) const {
        return (lhs - rhs)*(lhs - rhs);
    }
};

#endif
