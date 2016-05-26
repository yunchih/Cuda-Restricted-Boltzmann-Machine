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
#include <algorithm>
#include <random>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <ctime>
#include "utils.h"
#include "blas.h"
#include "mnist_reader.h"

class RBM {

public:
    RBM(int _n_visible, int _n_hidden, float _learning_rate, int _n_epoch, MnistReader& _reader);
    ~RBM();
    void train();

private:
    void update_w(const float* h_0, const float* v_0, const float* h_k, const float* v_k);
    void update_b(const float* v_0, const float* v_k);
    void update_c(const float* h_0, const float* h_k);
    void do_contrastive_divergence(const float* v_0);
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
    int n_epoch, n_train_data;
    float learning_rate;

    /* pW: _n_visible * _n_hidden matrix */
    /* pb: _n_visible vector */
    /* pc: _n_hidden vector  */
    float *pW, *pb, *pc;

    MnistReader reader;
    Blas blas;
};

#endif
