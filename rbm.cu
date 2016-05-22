#include "rbm.h"
#include "utils.h"

template <bool do_sample>
__global__ void add_sigmoid(float* x, const float* y, int size, curandState* rngs);
__global__ void add_diff(float* a, const float* x, const float* y, const float c, int size);
__forceinline__ __device__ float sample( float v, curandState* globalState );
__forceinline__ __device__ float sigmoidf(float in);

RBM::RBM(int _n_visible, int _n_hidden, float _learning_rate, int _n_cd_iter, int _n_epoch, int _minibatch_size):
    n_visible(_n_visible), n_hidden(_n_hidden), learning_rate(_learning_rate),
    n_cd_iter(_n_cd_iter), n_epoch(_n_epoch), minibatch_size(_minibatch_size){

    W = thrust::device_malloc<float>(_n_visible * _n_hidden);
    b = thrust::device_malloc<float>(_n_visible);
    c = thrust::device_malloc<float>(_n_hidden);
    pW = thrust::raw_pointer_cast(W);
    pb = thrust::raw_pointer_cast(b);
    pc = thrust::raw_pointer_cast(c);
    randn(pW, _n_visible * _n_hidden);
    randn(pb, _n_visible);
    randn(pc, _n_hidden);

    // Random number states used in sampling hidden/visible states
    cudaMalloc((void**)&rngs, std::max(_n_visible, _n_hidden) * sizeof(curandState));
    setup_random_numbers<<<1,std::max(_n_visible, _n_hidden)>>>(rngs, time(NULL));
}

void RBM::update_w(const float* h_0, const float* v_0, const float* h_k, const float* v_k){
    // W += learning_rate * (outer(h_0, v_0) - outer(h_k, v_k))
    add_outer_prod(this->pW, h_0, v_0, n_hidden, n_visible,  learning_rate);
    add_outer_prod(this->pW, h_k, v_k, n_hidden, n_visible, -learning_rate);
}
void RBM::update_b(const float* v_0, const float* v_k){
    // b += learning_rate * (v_0 - v_k)
    const int bsize = 128;
    const int gsize = CeilDiv(n_visible,bsize);
    add_diff<<<gsize,bsize>>>(this->pb, v_0, v_k, learning_rate, n_visible);
}
void RBM::update_c(const float* h_0, const float* h_k){
    // c += learning_rate * (h_0 - h_k)
    const int bsize = 128;
    const int gsize = CeilDiv(n_hidden,bsize);
    add_diff<<<gsize,bsize>>>(this->pc, h_0, h_k, learning_rate, n_hidden);
}
void RBM::do_contrastive_divergence(const float* v_0){
    static float *v_k = NULL, *h_k = NULL, *h_0 = NULL;
    if(v_k == NULL){
        cudaMalloc((void**) &v_k, sizeof(float)*n_visible);
        cudaMalloc((void**) &h_k, sizeof(float)*n_hidden);
        cudaMalloc((void**) &h_0, sizeof(float)*n_hidden);
    }

    // Initial hidden unit sampling: h0
    get_h_given_v<true>( h_0, v_0 );
    get_v_given_h<true>( h_k, v_k );

    cudaMemcpy(v_k, v_0, sizeof(float)*n_visible, cudaMemcpyDeviceToDevice);

    // Gibbs sampling
    for(int i = 0; i < this->n_cd_iter; ++i){
        get_h_given_v<true>( h_k, v_k );
        get_v_given_h<true>( h_k, v_k );
    }

    get_h_given_v<false>( h_k, v_k );

    this->update_w( h_0, v_0, h_k, v_k );
    this->update_b( v_0, v_k );
    this->update_c( h_0, h_k );
}
void RBM::train(float* training_data, int train_data_size){
    for(int i = 0; i < minibatch_index + minibatch_size; ++i){
        do_contrastive_divergence(training_data[i]);
        /* calculate cost here */
    }
}
void RBM::init_train_data(std::vector<float*> train_data){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (train_data.begin(), train_data.end(), std::default_random_engine(seed));
}

__global__ void add_diff(float* a, const float* x, const float* y, const float c, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size )
        a[i] += c*(x[i] - y[i]);
}

template <bool do_sample>
void RBM::get_h_given_v(float* h, const float* v){
    // h = sigmoid(dot(v, W) + c)
    matrix_mul(v, this->pW, h, 1, n_visible, n_visible, n_hidden, n_hidden);
    const int bsize = 128;
    const int gsize = CeilDiv(n_hidden,bsize);
    add_sigmoid<do_sample><<<gsize,bsize>>>(h, this->pc, n_hidden, this->rngs);
}

template <bool do_sample>
void RBM::get_v_given_h(const float* h, float* v){
    // v = sigmoid(dot(h, W) + b)
    /* Transpose the second matrix */
    matrix_mul_tranpose_first(h, this->pW, v, 1, n_hidden, n_visible, n_hidden, n_visible); 
    const int bsize = 128;
    const int gsize = CeilDiv(n_visible,bsize);
    add_sigmoid<do_sample><<<gsize,bsize>>>(v, this->pb, n_visible, this->rngs);
}

/*
 * =-=-=-=-=-=-=-=-=-=-=-=-=-=
 *   Helper functions below
 * =-=-=-=-=-=-=-=-=-=-=-=-=-=
 */

// x = sigmoid(x + y)
template <bool do_sample>
__global__ void add_sigmoid(float* x, const float* y, int size, curandState* rngs){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size ){
        float v = sigmoidf(x[i] + y[i]);
        x[i] = do_sample ? sample(v, &(rngs[i])) : v;
    }
}

/*
 * Sample a Bernoulli sample, return (0.0 or 1.0)
 */
__forceinline__ __device__ float sample( float v, curandState* globalState ) {
    int x = threadIdx.x;
    curandState localState = globalState[x];
    float rand = curand_uniform(&localState);
    globalState[x] = localState; 
    return rand > v ? 0.0 : 1.0;
}

/*
 * Compute the Sigmoid function
 */
__forceinline__ __device__ float sigmoidf(float in) {
    // raw approximation to sigmoid function
    return in / (1.f + fabsf(in));  
}
