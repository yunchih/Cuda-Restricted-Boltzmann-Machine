# Cuda Restricted Boltzmann Machine

## What is Restricted Boltzmann Machine
**Restricted Boltzmann Machine**(RBM) is a unsupervised learning algorithm(or, a generative model) invented by Paul Smolensky in 1986 and rediscovered by Geoffrey Hinton, who made RBM an useful neural network basis for larger modern machine learning model, such as **Deep Belief Network**.  Recent work on DBN: [Andrew Ng](https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf) and [Geoffrey Hinton](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5704567&tag=1). 

RBM is a variant of Boltzmann Machine which has a bipartite structure like the following graph[1]:

![RBM structure](http://deeplearning4j.org/img/sym_bipartite_graph_RBM.png)

Since RBM has the shape of a bipartite graph, with no intra-layer connections, the hidden unit activations are mutually independent given the visible unit activations and conversely, the visible unit activations are mutually independent given the hidden unit activations[2].  Thus, the activation functions are simply:

![RBM visible unit update function](https://en.wikipedia.org/api/rest_v1/media/math/render/svg/8df071af31d840d426ae5f4cfc111dd77b22770b)

and

![RBM hidden unit update function](https://en.wikipedia.org/api/rest_v1/media/math/render/svg/057f0c5b5e369ebac4ecc1053a7fcec0af48567d)

## Why a Cuda implementation
The nature of RBM's update rule makes it ideal for parallelization.  For example, the visible/hidden unit update functions above can be reduced to one matrix multiplication and a vector addition, both can be accelerated by GPU.

The implementation uses **[CuBLAS](https://developer.nvidia.com/cublas)** for matrix multiplication and matrix transpose, which is claimed to deliver 6x to 17x faster performance than the latest MKL BLAS, **[CuRAND](https://developer.nvidia.com/curand)** for random number generation, also claimed to be 75x faster than its Intel counterpart.  In addition, native Cuda kernels are implemented for Sigmoid function, vector addition, etc.

## Build

* Build: `make`

## Run

```

./rbm [Output directory] [Train filename] [Test filename] [Learning rate] [Epoch number] [CD step] [Train data size] [Test data size] [Random sample size]

```

#### Sample run 

1. `mkdir data`
2. extract mnist training set, i.e. `train-images-idx3-ubyte`, into `data` directory
2. `make runall`

#### Argument

* **Output directory**: directory where output filter images be stored
* **Train filename**: MNIST training data, i.e. `train-images-idx3-ubyte`
* **Test filename**: MNIST testing data, i.e. `t10k-images-idx3-ubyte`
* **Learning rate** 
* **Epoch number** 
* **CD step**: number of steps taken in Constrastive Divergence
* **Train data size**
* **Test data size**
* **Random sample size**: number of samples used to estimate the error of reconstruction in each epoch

### Development platform

* Cuda 6.5
* gcc 4.8.5
* Ubuntu 15.10 4.2.0-34-generic
* ImageMagick 6.8.9

### Prerequisite

* ImageMagick
* Cuda Runtime

### Reference
* [1] http://deeplearning4j.org
* [2] https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine 

### Also see

* [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
* [An Introductio to Restricted Boltzmann Machines](http://image.diku.dk/igel/paper/AItRBM-proof.pdf)
* [deeplearning.net](http://deeplearning.net/tutorial/rbm.html)
* [deep belief network by albertbup](https://github.com/albertbup/deep-belief-network)

#
