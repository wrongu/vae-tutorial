from vae import VAE
from latents import DiagonalGaussianLatent
from priors import IsoGaussianPrior
from likelihoods import DiagonalGaussianLikelihood
from keras.layers import Input, Dense
from data.my_mnist import img_pixels as mnist_pixels
import os


def fit_vae(vae, x_train, x_test=None, epochs=100, batch=100, weights_file=None, recompute=False, optimizer='adam'):
    """Fit a vae object to the given dataset (for datasets that fit in memory). Both x_train and
       x_test must have a number of data points divisible by the batch size.
    """

    # Load existing weights if they exist
    if weights_file is not None:
        if os.path.exists(weights_file) and not recompute:
            vae.model.load_weights(weights_file)
            return vae

    # Train the model
    vae.model.compile(loss=None, optimizer=optimizer)
    if x_test is not None:
        kwargs = {'validation_data': (x_test, None)}
    else:
        kwargs = {}
    vae.model.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch, **kwargs)

    # Save trained model to a file if given
    if weights_file is not None:
        vae.model.save_weights(weights_file)


def gaussian_mnist(latent_dim=2, pixel_std=.05):
    # Create and return and instance of the VAE class for MNIST digits.
    # YOUR CODE HERE
    inpt = Input(shape=(mnist_pixels,))
    # ...create the Q model here
    # latent = ...  # should have dimensionality = latent_dim
    # ...create the P model here
    # reconstruction = ...
    # likelihood = ...  # should be gaussian with mean=reconstruction and std=pixel_std
    return VAE(inpt, latent, reconstruction, likelihood)
