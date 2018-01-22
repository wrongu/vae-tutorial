Exercise 02 - Extending the ELBO with an analytic KL term
=========================================================

A Generic Latent Base-Class
---------------------------

In the first exercise, we saw one latent class called `DiagonalGaussianLatent` in `latents.py`.  In this exercise, a generic base class has been created called `Latent` which now takes care of the constructor and computing the output shape, since these operations would be common to all `Latent`s. Each `Latent` subclass must still implement the following as part of the keras `Layer` interface:

* `build()` - creates layer parameters
* `call(x)` - implements the layer's actual transformation of inputs to outputs

In the previous exerceise we added functions specific to our definition of a `Latent`, namely:

* `log_prob(x)` - uses the current/most recent parameters to compute the log probability of a batch of inputs
* `sample_kl()` - uses the current/most recent _sample_ of the latent to compute a monte-carlo estimate of KL from its posterior to the prior.

Implementing an analytic KL method for gaussians
------------------------------------------------

For certain distributions, the KL term in the ELBO objective can be computed analytically. In exercise 01, we used the fact that KL is an _expectation_ to estimate KL using samples of `Q` (the distribution defined by `self.mean` and `self.log_var` in the `DiagonalGaussianLatent` class). This estimate of KL will have high variance simply by virtue of being a monte-carlo estimate. Because our current prior is also Gaussian, we can instead use the following formula for KL between two gaussians:

  kl(p1||p2) = [log(det(C2)/det(C1)) - dim + Tr(C2^-1*C1) + (m2-m1).T*C2^-1*(m2-m1)]/2

where `C1` and `C2` are covariances, `m1` and `m2` are means, `det` is the determinant, and `Tr` is the trace.

**The goal of this exercise is to implement an interface where the ELBO objective uses the analytic form of KL when it is available and automatically falls back to the monte-carlo estimate when it is not.** For example, if we later choose to replace the gaussian prior with some complicated nonparametric form, it would automatically fall back to the monte carlo estimate under the hood with no extra work in designing the model.

**Implement `DiagonalGaussianLatent.analytic_kl()`.** Just like `sample_kl()`, it takes no inputs but instead uses the current values in `self.mean`, `self.log_var`, and `self.prior`. Your function should return a keras tensor with shape `(batch,)`. Hint: the `IsoGaussianPrior` class has mean `0` and covariance equal to the identity matrix. Using this, you should be able to compute KL using only `K.exp` and `K.sum`.

If `self.prior` is not an instance of a class for which the analytic form is known, your `analytic_kl` method should raise a `TypeError` (in python you can check if `isinstance(self.prior, IsoGaussianPrior)`).

Flexibly choosing between analytic and monte-carlo KL
-----------------------------------------------------

**Implement the "fallback" logic in `VAE.elbo`.** Since `analytic_kl` throws a `TypeError`, you can implement the "fall back" logic using `try: ... except TypeError: ...`.

You may be worried that `try ... except ...` is either inelegant or slow. Regarding elegance, it is a surprisingly common pattern to see in python. Regarding speed, remember that _the `elbo` function is only ever called once._ This is the key difference between working with computation graphs like tensorflow or theano and working directly with data in numpy. Since this is a computation graph, `VAE.elbo()` simply builds a series of operations that are not _executed_ until later, so speed while building the operations is never a concern!

Train a model
-------------

To train the model, `cd` into `02 - ELBO with analytic KL` and run `python run.py`.

Bonus exercise(s)
-----------------

Compare the training time to reach a certain loss using the different KL methods.
