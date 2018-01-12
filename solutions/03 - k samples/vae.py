import keras.backend as K
from keras.engine import Model


class VAE(object):
    def __init__(self, inpt, latent, reconstruction, likelihood, k_samples):
        # Create self.inpt, self.latent, self.reconstruction, and self.likelihood
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

        # KL term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch,) if analytic or
        # (batch, samples) otherwise
        try:
            # Use analytic KL if it is available, which has shape (batch,)
            self.kl = self.latent.analytic_kl()

            return K.sum(self.ll, axis=-1) / K.cast(self.k_samples, 'float32') - self.kl
        except TypeError:
            # If analytic KL is not available, fall back on sample KL.
            self.kl = self.latent.sample_kl()

            # ELBO is mean-over-samples of (LL - KL) and has shape (batch,)
            return K.sum(self.ll - self.kl, axis=-1) / K.cast(self.k_samples, 'float32')
