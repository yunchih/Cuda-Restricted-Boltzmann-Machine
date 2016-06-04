# Cuda Restricted Boltzmann Machine

# Build

* Build: `make`

## Run

```

./rbm [Output directory] [Train filename] [Test filename] [Learning rate] [Epoch number] [CD step] [Train data size] [Test data size] [Random sample size]

```

#### Sample run 

1. `mkdir data`
2. extract mnist training set, i.e. `train-images-idx3-ubyte`, into `data` directory
2. `make run`

### Argument

* **Output directory**: directory where output filter images be stored
* **Train filename**: MNIST training data, i.e. `train-images-idx3-ubyte`
* **Test filename**: MNIST testing data, i.e. `t10k-images-idx3-ubyte`
* **Learning rate** 
* **Epoch number** 
* **CD step**: number of steps taken in Constrastive Divergence
* **Train data size**
* **Test data size**
* **Random sample size**: number of samples used to estimate the error of reconstruction in each epoch

# Development platform

* Cuda 6.5
* gcc 4.8.5
* Ubuntu 15.10 4.2.0-34-generic
* ImageMagick 6.8.9

# Prerequisite

* ImageMagick
* Cuda Runtime

# Reference

* [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
* [An Introductio to Restricted Boltzmann Machines](http://image.diku.dk/igel/paper/AItRBM-proof.pdf)
* [deeplearning.net](http://deeplearning.net/tutorial/rbm.html)
* [deep belief network by albertbup](https://github.com/albertbup/deep-belief-network)

