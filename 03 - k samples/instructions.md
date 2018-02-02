Exercise 03 - K samples per latent
==================================

This is probably the most complicated exercise simply due to the fact managing array sizes in Keras can be tricky.

The goal of this exercise is to improve the Monte-Carlo estimates we've been using so far by drawing multiple samples of the latents per training point rather than the single sample we've been using so far. Further, the number of samples drawn will be a variable in the computation graph so that it can be manipulated on the fly. Both instances of `Latent` classes and instances of the `VAE` class will need to share the same variable for the number of latent samples.

Modify the `Latent` class I: output multiple samples of the latent
-------------------------

First, notice that the `Latent` base class now takes a third argument in its constructor called `k_samples`. Expect this to be a integer-valued keras variable.

One goal of this implementation is for the rest of the model to be independent of the number of samples. That is, we shouldn't have to change the architecture of the generative/`p` model or the recognition/`q` model to use a different number of samples. The generative model currently expects input of size `(batch x latent_dim)`, where `batch` is the number of input digits being processed at once in MNIST, for example. We now want `k_samples` per latent variable per input image. It will convenient to store samples in a size `(batch x k_samples x latent_dim)` array for some purposes, but for the generative model component we will want to "flatten" this into something of size `(??? x latent_dim)`. A Keras layer is provided that does this called `FoldSamplesIntoBatch`.

![Visualization of how the `FoldSamplesIntoBatch` layer affects array sizes by reshaping an array of size `(batch, samples, ...)` into an array of size `(batch * samples, ...)`](../images/flatten_batch_visualization.png)

**Update `DiagonalGaussianLatent.call()`** to draw `k_samples` samples per latent variable (per `latent_dim`, that is). Store `(batch x samples x latent_dim)` result in `self.samples`, and call a `FoldSamplesIntoBatch` layer to get a reshaped copy. (Note: if we were to assume `k_samples` were a fixed quantity, this would be much easier. Dynamically resizing the batch dimension of arrays is not really _expected_ use of keras, hence the need for an entire helper class.)

Modify the `Latent` class II: update `log_prob`
-------------------------

**Update `DiagonalGaussianLatent.log_prob()`** to expect inputs of size `(batch x samples x latent_dim)` and output log probabilities of size `(batch x samples)` using the corresponding mean and log variances per batch.

Modify the `Latent` class III: update `sample_kl`
-------------------------

**Update `DiagonalGaussianLatent.sample_kl()`** to use all of the `k` samples in its monte carlo estimate. A useful function will be `K.cast`.

Note that `analytic_kl` does not need to be updated!

Modify the `VAE` class
----------------------

Since the inputs to the recognition model and outputs from the generative model will now have different sizes (there will be `k` different reconstructions per input), you will need to update the `VAE.elbo()` function, specifically the way the log-likelihood is used. The final likelihood _per input_ will be the expected value over sampled reconstructions. Useful functions will be `K.batch_flatten`, `K.repeat`, and `K.reshape`.

Update `models.py`
------------------

The above design has the `VAE` and `Latent` classes all sharing a reference to the same underlying `k_samples` variable. This will make it easy later to do things like start training quickly initially with a small number of samples and refine it later with a single call to `vae.set_samples()`, updating both the sampling and loss-computation aspects of the model.

This single, shared variable `k_samples` must now be created in `models.py` and passed to both the `Latent` constructor and the `VAE` constructor.

Train a model
-------------

To train the model, `cd` into `03 - k samples` and run `python run.py`. Try it with different numbers of samples (modify `run.py` to save different `weights.h5` files for each one).

Bonus exercise(s)
-----------------

Implement a training schedule that increases `k_samples` every few epochs.
