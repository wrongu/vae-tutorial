import keras.backend as K
from keras.engine import Model


class IWAE(object):
    def __init__(self, inpt, latents, reconstruction, likelihood, k_samples):
        # Create self.inpt, self.latents, self.reconstruction, and self.likelihood
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object.
        self.model = Model(inpt, reconstruction)
        self.model.add_loss(-self.elbo())

    def set_samples(self, k):
        K.set_value(self.k_samples, k)

    def elbo(self):
        batch = K.shape(self.inpt)[0]
        # shape of flat_input is (batch, pixels)
        flat_input = K.batch_flatten(self.inpt)
        # shape of repeated_input is (batch, samples, pixels)
        repeated_input = K.repeat(flat_input, self.k_samples)
        # shape of flat_repeated_input is (batch * samples, pixels) to match the shape of self.reconstruction
        flat_repeated_input = K.reshape(repeated_input, (batch * self.k_samples, -1))

        # LL term is E_q(z|x) [ log p(x|z) ] (Note that mean over k_samples happens later)
        # shape of flat_ll is (batch * samples,).
        flat_ll = self.likelihood.log_prob(flat_repeated_input)
        # shape of self.ll is (batch, samples)
        self.ll = K.reshape(flat_ll, (batch, -1))

        # Final loss is weighted sum across k samples. More precisely, the total gradient is a
        # weighted sum of sample gradients. K.stop_gradient() is used to make the weights act on
        # gradients and not provide gradients themselves (weights are not 'learned' per se).
        # Weights have shape (batch, samples).
        weights = K.stop_gradient(self._get_weights())

        # KL term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch,) if analytic or (batch,
        # samples) otherwise. One KL term per latent. Each may be analytic or monte carlo.
        self.kl_terms = [None] * len(self.latents)

        for i, l in enumerate(self.latents):
            try:
                # Use analytic KL if it is available, which has shape (batch,)
                self.kl_terms[i] = l.analytic_kl()
            except TypeError:
                # If analytic KL is not available, fall back on sample KL.
                self.kl_terms[i] = K.sum(weights * l.sample_kl(), axis=-1)

        # ELBO is mean-over-samples of (LL - KL) and has shape (batch,)
        return K.sum(weights * self.ll, axis=-1) - K.sum(self.kl_terms, axis=0)

    def _get_weights(self):
        # IWAE sample weight on sample i is p(x,latent_i)/q(latent_i|x). Weights are then
        # normalized to sum to 1. First computing log-weights is more numerically stable.
        log_p = K.sum([K.reshape(l.prior.log_prob(l.flat_samples), (-1, self.k_samples, l.dim))
                       for l in self.latents], axis=0) + self.ll
        log_q = K.sum([l.log_prob(l.samples) for l in self.latents], axis=0)
        log_weights = log_p - log_q
        # Pre-normalize results in log space by subtracting logsumexp (which is dividing by sum in
        # probability space). This keeps results stable so that the following call to exp() is
        # given values in a reasonable range. Results may not sum to exactly 1, though, due to
        # floating point precision.
        log_weights -= K.logsumexp(log_weights, axis=-1, keepdims=True)
        # Get out of log space and normalize a second time since logsumexp is not perfect.
        weights_unnormalized = K.exp(log_weights)
        return weights_unnormalized / K.sum(weights_unnormalized, axis=-1, keepdims=True)


class VAE(IWAE):
    def _get_weights(self):
        # Return 1/k equal weight on all samples with shape (batch, samples).
        weights_unnormalized = K.ones_like(self.latents[0].samples[:, :, 0])
        return weights_unnormalized / K.cast(self.k_samples, 'float32')
