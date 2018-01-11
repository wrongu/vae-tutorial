import keras.backend as K
from keras.engine import Model


class VAE(object):
    def __init__(self, inpt, latent, reconstruction, likelihood):
        # Create self.inpt, self.latent, self.reconstruction, and self.likelihood
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object.
        self.model = Model(inpt, reconstruction)
        self.model.add_loss(self.elbo_loss())

    def elbo_loss(self):
        flat_input = K.batch_flatten(self.inpt)

        # NLL loss term is E_q(z|x) [ -log p(x|z) ] and has shape (batch,)
        self.nll = -self.likelihood.log_prob(flat_input)

        # KL loss term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch,)
        kl_loss = self.latent.kl()

        # Total loss is simply sum of KL and NLL terms and has shape (batch,)
        return kl_loss + self.nll
