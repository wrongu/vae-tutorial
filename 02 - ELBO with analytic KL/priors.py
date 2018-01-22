import keras.backend as K


class Prior(object):
    def __init__(self, dim):
        self.d = dim


class IsoGaussianPrior(Prior):
    def log_prob(self, x):
        return -K.sum(x * x, axis=-1) / 2

    def sample(self, n):
        return K.random_normal(shape=(n, self.d))
