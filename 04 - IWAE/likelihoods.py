from keras.engine.topology import Layer
import keras.backend as K


class DiagonalGaussianLikelihood(Layer):

    def __init__(self, mean, std):
        # Mean must be a vector of shape (dim,). std may be a vector of the same shape or a scalar.
        self.mean = mean
        # If std is a scalar, this creates an array of [var, var, var, ...]. If it is already a
        # vector, this does nothing.
        self.var = K.ones_like(mean) * (std ** 2)

    def log_prob(self, x):
        # Determinant of the diagonal covariance matrix is the product of variances.
        log_det = K.sum(K.log(self.var))
        return -K.sum(K.square(x - self.mean) / (2 * self.var), axis=-1) - log_det / 2
