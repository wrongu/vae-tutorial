import keras.backend as K
from keras.engine import Model


class VAE(object):
    def __init__(self, inpt, latent, reconstruction, likelihood):
        # Create self.inpt, self.latent, self.reconstruction, and self.likelihood
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object.
        self.model = Model(inpt, reconstruction)
        # To maximize ELBO, keras will minimize "loss" of -ELBO
        self.model.add_loss(-self.elbo())

    def elbo(self):
        flat_input = K.batch_flatten(self.inpt)

        # LL term is E_q(z|x) [ log p(x|z) ] and has shape (batch,)
        self.ll = self.likelihood.log_prob(flat_input)

        # YOUR CODE HERE
        # self.kl = ...

        # ELBO simply (LL - KL) and has shape (batch,)
        return self.ll - self.kl
