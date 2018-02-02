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
        # YOUR CODE HERE
