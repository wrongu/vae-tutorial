import keras.backend as K
from keras.engine import Model


class VAE(object):
    def __init__(self, inpt, latent, reconstruction, likelihood, k_samples):
        # Create self.inpt, self.latent, self.reconstruction, and self.likelihood
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object.
        self.model = Model(inpt, reconstruction)
        self.model.add_loss(self.elbo_loss())

    def set_samples(self, k):
        K.set_value(self.k_samples, k)

    def elbo_loss(self):
        batch = K.shape(self.inpt)[0]
        # shape of flat_input is (batch, pixels)
        flat_input = K.batch_flatten(self.inpt)
        # shape of repeated_input is (batch, samples, pixels)
        repeated_input = K.repeat(flat_input, self.k_samples)
        # shape of flat_repeated_input is (batch * samples, pixels) to match the shape of self.reconstruction
        flat_repeated_input = K.reshape(repeated_input, (batch * self.k_samples, -1))

        # NLL loss term is E_q(z|x) [ -log p(x|z) ] (Note that division by k_samples happens later)
        # shape of flat_nll is (batch * samples,).
        flat_nll = -self.likelihood.log_prob(flat_repeated_input)
        # shape of self.nll is (batch, samples)
        self.nll = K.reshape(flat_nll, (batch, -1))

        try:
            # Use analytic KL if it is available, which has shape (batch,)
            kl_loss = self.latent.analytic_kl()

            return kl_loss + K.sum(self.nll, axis=-1) / K.cast(self.k_samples, 'float32')
        except TypeError:
            # If analytic KL is not available, fall back on sample KL.
            kl_loss = self.latent.sample_kl()

            # Total loss per input is mean-over-samples of KL and NLL terms and has shape (batch,)
            return K.sum(kl_loss + self.nll, axis=-1) / K.cast(self.k_samples, 'float32')
