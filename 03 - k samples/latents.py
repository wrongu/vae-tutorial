from keras.engine.topology import Layer
from priors import IsoGaussianPrior
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

        # YOUR CODE HERE
        # self.samples = ...
        # self.flat_samples = ...

        return self.flat_samples

    def log_prob(self, x):
        """Given batch of x of shape (batch, samples, dim), returns (batch, samples) values of the
           log probability per sample.
        """
        # YOUR CODE HERE

    def sample_kl(self):
        # YOUR CODE HERE

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
