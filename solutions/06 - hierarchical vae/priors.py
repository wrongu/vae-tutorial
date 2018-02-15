import keras.backend as K
import numpy as np


class Prior(object):
    def __init__(self, dim):
        self.d = dim


class IsoGaussianPrior(Prior):
    def log_prob(self, x):
        return -K.sum(x * x, axis=-1) / 2

    def sample(self, n):
        return K.random_normal(shape=(n, self.d))


class DiagonalGaussianPrior(Prior):
    def __init__(self, dim, mean, log_var):
        self.d = dim
        self.mean = mean
        self.log_var = log_var
        self.var = K.exp(self.log_var)
        self.std = K.exp(self.log_var / 2)

    def log_prob(self, x):
        log_partition = -(np.log(2 * np.pi) + K.sum(self.log_var, axis=-1)) / 2
        return -K.sum((x - self.mean)**2 / self.var, axis=-1) / 2 + log_partition
