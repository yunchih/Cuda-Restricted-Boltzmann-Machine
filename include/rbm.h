#ifndef RBM_H
#define RBM_H

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <algorithm>
#include <random>
#include <chrono>  
#include "utils.h"

#define CHECK {\
    auto e = cudaDeviceSynchronize();\
    if (e != cudaSuccess) {\
        printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
        abort(); \
    }\
}

class RBM {


public:
    RBM(int _n_visible, int _n_hidden, float _learning_rate, int _n_cd_iter, int _n_epoch, int _minibatch_size);
    void train(float* training_data, int total);

private:
    void update_w(const float* h_0, const float* v_0, const float* h_k, const float* v_k);
    void update_b(const float* v_0, const float* v_k);
    void update_c(const float* h_0, const float* h_k);
    void do_contrastive_divergence(const float* v_0);
    void init_train_data(std::vector<float*> train_data);
    void sample_h(float* h_s, const float* h_0);

    template <bool do_sample>
    float* get_h_given_v(float* h, const float* v);

    template <bool do_sample>
    float* get_v_given_h(const float* h, float* v);


/* Fields */
    int n_visible, n_hidden;
    int n_cd_iter, n_epoch, minibatch_size;
    float learning_rate;

    /* state variables needed for random number generation */
    curandState* rngs;

    /* pW: _n_visible * _n_hidden matrix */
    /* pb: _n_visible vector */
    /* pc: _n_hidden vector  */
    float *pW, *pb, *pc;
};

#endif
