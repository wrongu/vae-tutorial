from keras.engine.topology import Layer
from priors import IsoGaussianPrior, DiagonalGaussianPrior
import keras.backend as K


class Latent(Layer):
    """Base class for VAE latents.
    """

    def __init__(self, dim, prior, k_samples, **kwargs):
        # Call Layer constructor
        super(Latent, self).__init__(**kwargs)

        # Record instance variables
        self.dim = dim
        self.prior = prior
        self.k_samples = k_samples

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.dim,)


class DiagonalGaussianLatent(Latent):
    """DiagonalGaussianLatent expects flattened input with shape (batch, dim). Internally stores
       2*d parameters: 'mean' and 'log_var' of each dimension of the posterior distribution that
       are themselves each constructed as a dense connection from inputs. Output is (batch, d)
       *sampled value* of each latent, where d is the dimensionality passed to the constructor.
    """

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
        sample_shape = (K.shape(self.mean)[0], self.k_samples, self.dim)
        eps = K.random_normal(shape=sample_shape, mean=0., stddev=1.0)

        # Shape of self.samples is (batch, k, dim) - need to repeat mean and std k times
        self.samples = K.repeat(self.mean, self.k_samples) + eps * K.repeat(std, self.k_samples)

        # Shape of self.flat_samples is (batch * k, dim) and can be fed directly into the
        # generative model.
        self.flat_samples = FoldSamplesIntoBatch()(self.samples)

        return self.flat_samples

    def log_prob(self, x):
        """Given batch of x of shape (batch, samples, dim), returns (batch, samples) values of the
           log probability per sample.
        """
        # log gaussian probability = -1/2 sum[(x-mean)^2/variance]
        variance = K.repeat(K.exp(self.log_var), self.k_samples)  # shape is (batch, samples, dim)
        log_det = K.tile(K.sum(self.log_var, axis=-1, keepdims=True), (1, self.k_samples))  # shape is (batch, samples)
        x_diff = x - K.repeat(self.mean, self.k_samples)  # shape is (batch, samples, dim)
        return -(K.sum((x_diff / variance) * x_diff, axis=-1) + log_det) / 2

    def sample_kl(self):
        # Monte carlo KL estimate is the expected value over samples of self.log_prob - prior.log_prob
        log_prior = K.reshape(self.prior.log_prob(self.flat_samples), (-1, self.k_samples))
        return (self.log_prob(self.samples) - log_prior) / K.cast(self.k_samples, 'float32')

    def analytic_kl(self):
        if isinstance(self.prior, IsoGaussianPrior):
            # In general for two multi-variate normals
            #   kl(p1||p2)=[log(det(C2)/det(C1)) - dim + Tr(C2^-1*C1) + (m2-m1).T*C2^-1*(m2-m1)]/2
            # where C1 and C2 are covariances, and m1 and m2 are means. Since 'IsoGaussianPrior' is
            # mean 0 and identity covariance, this is simplified significantly:
            #   kl(p1||iso)=[-log(det(C1)) - dim + Tr(C1) + m1.T*m1]/2
            log_det_p1 = K.sum(self.log_var, axis=-1)
            trace_c1 = K.sum(K.exp(self.log_var), axis=-1)
            mean_sq_norm = K.sum(self.mean**2, axis=-1)
            return (-log_det_p1 - self.dim + trace_c1 + mean_sq_norm) / 2
        else:
            raise TypeError("Prior must be IsoGaussianPrior to use analytic_kl on DiagonalGaussianLatent")


class FoldSamplesIntoBatch(Layer):
    """Keras 'layer' that reshapes a (batch, K, ...) input into (batch * K, ...).

       This may not always play nicely with other parts of Keras, since there are some internal
       sanity-checks that sizes and shapes of things.
    """

    def call(self, x):
        input_shape = K.shape(x)

        # Multiply the first to dimensions together and leave the rest unchanged.
        new_shape = K.concatenate([[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
        return K.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            return (None,) + tuple(input_shape[2:])
        else:
            return (input_shape[0] * input_shape[1],) + tuple(input_shape[2:])
