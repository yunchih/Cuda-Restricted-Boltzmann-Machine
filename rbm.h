#ifndef RBM_H
#define RBM_H

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <algorithm>
#include <random>
#include <chrono>  

#define CHECK {\
    auto e = cudaDeviceSynchronize();\
    if (e != cudaSuccess) {\
        printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
        abort(); \
    }\
}

class RBM {

    int n_visible, n_hidden;
    int n_cd_iter, minibatch_size;
    float learning_rate;
    /* W: _n_visible * _n_hidden matrix */
    /* b: _n_visible vector */
    /* W: _n_hidden vector  */
    thrust::device_ptr<float> W, b, c;
    float *pW, *pb, *pc;

public:
    RBM(int _n_visible, int _n_hidden, float _learning_rate, int _n_cd_iter, int _n_epoch, int _minibatch_size):
        n_visible(_n_visible), n_hidden(_n_hidden), learning_rate(_learning_rate),
        n_cd_iter(_n_cd_iter), n_epoch(_n_epoch), minibatch_size(_minibatch_size);

};

#endif
