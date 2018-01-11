from keras.engine.topology import Layer
import keras.backend as K


class DiagonalGaussianLatent(Layer):
    """DiagonalGaussianLatent expects flattened input with shape (batch, dim). Internally stores
       2*d parameters: 'mean' and 'log_var' of each dimension of the posterior distribution that
       are themselves each constructed as a dense connection from inputs. Output is (batch, d)
       *sampled value* of each latent, where d is the dimensionality passed to the constructor.
    """

    def __init__(self, dim, prior, **kwargs):
        # Call Layer constructor
        super(DiagonalGaussianLatent, self).__init__(**kwargs)

        # Record instance variables
        self.dim = dim
        self.prior = prior

    def build(self, input_shape):
        # Create trainable weights of this layer for the two dense connections to 'mean' and to
        # 'log_var'.
        input_dim = input_shape[-1]
        self.dense_mean = self.add_weight(shape=(input_dim, self.dim),
                                          name='latent_mean_kernel',
                                          initializer='glorot_uniform')
        self.dense_log_var = self.add_weight(shape=(input_dim, self.dim),
                                             name='latent_log_var_kernel',
                                             initializer='glorot_uniform')
        self.built = True

    def call(self, x):
        # Apply matrix multiplication of inputs and the weights created in build() to get 'mean'
        # and 'log_var' parameters.
        self.mean = K.dot(x, self.dense_mean)
        self.log_var = K.dot(x, self.dense_log_var)

        # exp(log_var / 2) is standard deviation
        std = K.exp(self.log_var / 2)

        # Create (reparameterized) sample from the latent distribution
        sample_shape = (K.shape(self.mean)[0], self.dim)
        eps = K.random_normal(shape=sample_shape, mean=0., stddev=1.0)

        # Shape of self.sample is (batch, dim)
        self.sample = self.mean + eps * std

        return self.sample

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.dim,)

    def log_prob(self, x):
        # log gaussian probability = -1/2 sum[(x-mean)^2/variance]
        variance = K.exp(self.log_var)
        log_det = K.sum(self.log_var, axis=-1)
        x_diff = x - self.mean
        return -(K.sum((x_diff / variance) * x_diff, axis=-1) + log_det) / 2

    def sample_kl(self):
        # Monte carlo KL estimate is simply self.log_prob - prior.log_prob
        return self.log_prob(self.sample) - self.prior.log_prob(self.sample)
