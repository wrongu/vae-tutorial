from keras.engine.topology import Layer
import keras.backend as K


class DiagonalGaussianLikelihood(Layer):

    def __init__(self, mean, std):
        # Both mean and std should be scalars or tensors with shape (dim,)
        self.mean = mean
        self.var = std ** 2

    def log_prob(self, x):
        return -K.sum(K.square(x - self.mean) / (2 * self.var), axis=-1)
