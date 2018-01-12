from keras.engine.topology import Layer
import keras.backend as K


class DiagonalGaussianLatent(Layer):
    def __init__(self, dim, prior, **kwargs):
        # Call Layer constructor
        super(DiagonalGaussianLatent, self).__init__(**kwargs)

        # Record instance variables
        self.dim = dim
        self.prior = prior

    def build(self, input_shape):
        # Create trainable weights of this layer for the two dense connections to 'mean' and to
        # 'log_var'.
        # YOUR CODE HERE
        self.built = True

    def call(self, x):
        # Apply matrix multiplication of inputs and the weights created in build() to get 'mean'
        # and 'log_var' parameters.
        # YOUR CODE HERE
        # self.mean = ...
        # self.log_var = ...

        # Create (reparameterized) sample from the latent distribution
        # YOUR CODE HERE
        # self.sample = ...

        return self.sample

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.dim,)

    def log_prob(self, x):
        # Return the (batch,) log probabilities of x which has shape (batch, dim) using self.mean
        # and self.log_var. Note: make sure to take into account all terms that depend on mean and
        # log_var! (See likelihoods.py if you need help).
        # YOUR CODE HERE

    def sample_kl(self):
        # Return Monte Carlo estimate of KL to the prior using self.sample, self.prior, and self.log_prob
        # YOUR CODE HERE
