Exercise 01 - Implementing the ELBO objective and a simple VAE class
====================================================================

Files
-----

Since this is the first exercise, we will start by reviewing the contents of each file.

* run.py - the top-level script to be run from the command line
* vae.py - contains the definition of the `VAE` class
* models.py - functions for creating a `VAE` instance for MNIST and training a model
* latents.py - contains classes for VAE-style reparameterized latent variables
* priors.py - contains classes for prior distributions (over latent variables)
* likelihoods.py - contains classes for log-likelihood reconstruction error

The VAE class
-------------

Conceptually, a VAE contains two parts: the **recognition model** (`Q`) which maps from data to the _parameters_ of a distribution over latent variables, and the **generative model** (`P`) which maps from the latent variables back to the input space. In practice, we break these two parts down into sub-components and keep track of those:

    Input (flattened images)
       |
       |   Q model (neural net)
       V
    Latent parameters   <--   Prior
       |
       |
       V
    Latent sample(s)
       |
       |   P model (neural net)
       V
    Reconstruction

The role of the `VAE` class in `vae.py` is to wrap these components into a single object. That is, it keeps track of enough information to compute the ELBO objective. In this implementation, we will put some of the heavy-lifting into a `Latent` class to simplify the `VAE` class. For now, assume that `Latent` is handling the prior, drawing samples, and computing KL between the latents' posterior and the prior.

**Implement `VAE.elbo()`.** Use the following functions:

* `self.likelihood.log_prob(x)`, which will return the monte-carlo estimate of the log probability of `x` given the reconstruction.
* `self.latent.sample_kl()` which will return the monte-carlo estimate of the KL term described above.
* You may want to use `K.batch_flatten(self.inpt)` to ensure that the input is flattened.

The DiagonalGaussianLatent class
--------------------------------

We are keeping the implementation general, but focusing on a specific type of latent variable: a `dim`-dimensional gaussian random variable with diagonal covariance. In future exercises, we will create an abstract `Latent` base class, but for now `DiagonalGaussianLatent` is the only one.

Our `Latent` class will take `(batch, N)` flattened input, then effectively create two `Dense` layers projecting to a `self.mean` and `self.log_var` parameter for the `dim`-dimensional mean and log-variance of the posterior respectively. Using *log* variance is safe because it can be any real number, which should be easier for the neural net to produce.

`DiagonalGaussianLatent` extends the `Layer` class from keras. When writing custom layers, keras *requires* that the superclass constructor is called in `__init__`. It also requires a `call(self, x)` function and a `compute_output_shape(self, input_shape)` function. Finally, `build(self, input_shape)` is required for all layers that contain trainable weights. A skeleton of each of these functions is provided for you. `log_prob(self, x)` and `sample_kl(self)` are additional functions specific to a `Latent` for a VAE.

Notice that the constructor takes a `prior` as input. We will assume that `prior` is an instance of a class from `priors.py` and has a `log_prob(x)` function.

**Implement `DiagonalGaussianLatent.build`.** This should create two sets of trainable weights: one that projects from inputs to `self.mean` and one that projects from inputs to `self.log_var`. The actual matrix multiplication will happen later - for now, just create the weights.

* The relevant keras function is `Layer.add_weight(...)`, which you can access as `self.add_weight(...)` here.

**Implement `DiagonalGaussianLatent.call`.** This function determines what is actually produced as the output of this layer and passed as input to the next layer. It should do the following:

1. actually compute `self.mean` and `self.log_var` using the weights created in `build` and `K.dot()`
2. draw and store a _reparameterized sample_ of the latent variables using `K.random_normal` with a mean of `0` and `stddev` of `1`.
3. return the sample so it will then be passed into the next part of the model

**Implement `DiagonalGaussianLatent.log_prob(x)`.** This will return the log probability of (a batch of) data points `x` using `self.mean` and `self.log_var`. When used in a neural network, then, `log_prob(x)` will return different values for the same `x` depending on the inputs to the network, since these will affect `self.mean` and `self.log_var`. Useful functions:

* `K.exp()`
* `K.sum(..., axis=-1)`

**Implement `DiagonalGaussianLatent.sample_kl()`.** Recall that KL(Q||P) is defined as `E_Q[ log (Q(x) / P(x)) ]`, or equivalently `E_Q[ log Q(x) - log P(x) ]`. Since this instance of `Latent` defines `Q`, we already have a sample of `x` in `self.sample`. Use this to compute and return the monte-carlo estimate of the KL term that enters into the EBLO (this is not difficult and can be done in one line).

Build and train a model
-----------------------

This is the last step - open `models.py` and **implement the `gaussian_mnist` function.** This will combine all of the parts described above and return an instance of the `VAE` class. The relevant classes from keras have already been imported for you. In my solutions, I used the following architecture:

1. input is assumed to be flattened `(batch, mnist_pixels)` images
2. the Q model has two `Dense` layers with 64 units each and rectified linear activations
3. two-dimensional latent
4. the P model also has `Dense` layers with 64 units each and rectified linear activations, followed by a final `Dense` layer back up to full-pixel resolution with `mnist_pixels` outputs and sigmoid activation.

To train the model, `cd` into `01 - ELBO in keras` and run `python run.py`.

Bonus exercise(s)
-----------------

Here I have assumed a fixed value for `pixel_std=.05` per pixel, but in general the `P` model may output both a mean reconstruction and standard deviation _per pixel_, allowing it to specify some "confidence" about its own output. See `likelihoods.py` - it accepts a vector for `std`.

Try different numbers of layers and/or hidden units in the Q and P models.

Try changing setting `fit_vae(..., optimizer='sgd')`, or `optimizer='rmsprop'` in `run.py`

In `fit_vae`, create your own class that extends keras' [`EarlyStopping` callback](https://github.com/keras-team/keras/blob/76c5b616f2cac0d4d1852049d98ab8a067142373/keras/callbacks.py#L432) so that training automatically stops when validation loss stops improving. It's common when using keras to look inside their source code for help. (Note: you will need to implement your own since the standard `EarlyStopping` class monitors `val_loss` which is a vector, while we want it to monitor `np.mean(val_loss)`).
