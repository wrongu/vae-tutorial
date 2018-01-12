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
        # YOUR CODE HERE
        # Note that you have access to self.inpt, self.latent, self.reconstruction and
        # self.likelihood Expect self.inpt to have shape (batch, dim) (e.g. dim is 784 in MNIST).
        # Return a variable of size (batch,) - one value per input
