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
    # RECOGNITION MODEL
    inpt = Input(shape=(mnist_pixels,))
    q_hidden_1 = Dense(64, activation='relu')(inpt)
    q_hidden_2 = Dense(64, activation='relu')(q_hidden_1)

    # LATENT -- PRIOR
    latent = DiagonalGaussianLatent(dim=latent_dim, prior=IsoGaussianPrior(latent_dim))
    latent_sample = latent(q_hidden_2)

    # GENERATIVE MODEL
    gen_hidden_1 = Dense(64, activation='relu')(latent_sample)
    gen_hidden_2 = Dense(64, activation='relu')(gen_hidden_1)
    reconstruction = Dense(mnist_pixels, activation='sigmoid')(gen_hidden_2)

    # LIKELIHOOD
    # Note: in some models, pixel_std is not constant but is also an output of the model so that it
    # can indicate its own uncertainty.
    likelihood = DiagonalGaussianLikelihood(reconstruction, pixel_std)

    # Combine the above parts into a single model
    return VAE(inpt=inpt, latent=latent, reconstruction=reconstruction, likelihood=likelihood)
