from vae import VAE, IWAE
from latents import DiagonalGaussianLatent
from priors import IsoGaussianPrior, DiagonalGaussianPrior
from likelihoods import DiagonalGaussianLikelihood
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from data.my_mnist import img_pixels as mnist_pixels
from data.my_cifar10 import img_rows as cifar_rows
from data.my_cifar10 import img_cols as cifar_cols
from data.my_cifar10 import img_channels as cifar_channels
import keras.backend as K
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


def gaussian_mnist(cls, latent_dim=2, pixel_std=.05, k=1):
    # SINGLE 'SAMPLES' VARIABLE
    k_samples = K.variable(k, name='k_samples', dtype='int32')

    # RECOGNITION MODEL
    inpt = Input(shape=(mnist_pixels,))
    q_hidden_1 = Dense(64, activation='relu')(inpt)
    q_hidden_2 = Dense(64, activation='relu')(q_hidden_1)

    # LATENT -- PRIOR
    latent = DiagonalGaussianLatent(dim=latent_dim, prior=IsoGaussianPrior(latent_dim), k_samples=k_samples)
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
    if cls == 'vae':
        return VAE(inpt=inpt, latent=latent, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)
    elif cls == 'iwae':
        return IWAE(inpt=inpt, latent=latent, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)


def gaussian_mnist_hvae(cls, layers=3, latent_dim=8, pixel_std=.05, k=1):
    # SINGLE 'SAMPLES' VARIABLE
    k_samples = K.variable(k, name='k_samples', dtype='int32')

    latents = [None] * layers
    inpt = Input(shape=(mnist_pixels,))
    next_layer_in = inpt

    # RECOGNITION MODEL
    for l in range(layers):
        q_hidden = Dense(64, activation='relu')(next_layer_in)

        # LATENT
        # Note: prior will be set in second pass over layers since it depends on parent layer.
        latents[l] = DiagonalGaussianLatent(dim=latent_dim, prior=None, k_samples=k_samples)
        latents[l](q_hidden)

        # Input to next layer is means of this layer
        next_layer_in = latents[l].mean

    # GENERATIVE MODEL
    # Set prior of topmost layer to IsoGaussianPrior
    latents[-1].prior = IsoGaussianPrior(latent_dim)
    for l in reversed(range(layers - 1)):
        # Set prior of layer L as dense network from layer L+1's sampled values.
        gen_hidden = Dense(64, activation='relu')(latents[l + 1].flat_samples)

        mean = Dense(latent_dim)(latents[l + 1].flat_samples)
        log_var = Dense(latent_dim)(latents[l + 1].flat_samples)
        latents[l].prior = DiagonalGaussianPrior(latent_dim, mean=mean, log_var=log_var)

    # Reconstruct from the 0th layer
    gen_hidden = Dense(64, activation='relu')(latents[0].flat_samples)
    reconstruction = Dense(mnist_pixels, activation='sigmoid')(gen_hidden)

    # LIKELIHOOD
    # Note: in some models, pixel_std is not constant but is also an output of the model so that it
    # can indicate its own uncertainty.
    likelihood = DiagonalGaussianLikelihood(reconstruction, pixel_std)

    # Combine the above parts into a single model
    if cls == 'vae':
        return VAE(inpt=inpt, latents=latents, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)
    elif cls == 'iwae':
        return IWAE(inpt=inpt, latents=latents, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)


def gaussian_cnn_cifar10(cls, latent_dim=16, pixel_std=.05, k=1, filters=32, width=3):
    # SINGLE 'SAMPLES' VARIABLE
    k_samples = K.variable(k, name='k_samples', dtype='int32')

    # RECOGNITION MODEL
    inpt = Input(shape=(cifar_rows, cifar_cols, cifar_channels))
    q_hidden_1 = Conv2D(filters, kernel_size=width, activation='relu', padding='same')(inpt)
    q_hidden_2 = Conv2D(filters, kernel_size=width, activation='relu', padding='same')(q_hidden_1)
    q_hidden_flat = Flatten()(q_hidden_2)

    # LATENT -- PRIOR
    latent = DiagonalGaussianLatent(dim=latent_dim, prior=IsoGaussianPrior(latent_dim), k_samples=k_samples)
    latent_sample = latent(q_hidden_flat)

    # Determine how large reconstruction must be and insert a 'fan out' dense layer to return to
    # that size, without any nonlinearity.
    num_features = filters * cifar_rows * cifar_cols
    sample_fan_out = Dense(num_features)(latent_sample)
    sample_image = Reshape(target_shape=(cifar_rows, cifar_cols, filters))(sample_fan_out)

    # GENERATIVE MODEL Note that convolution is, in a sense, it's own inverse. That is, the
    # "deconvolution" counterpart of a 3x3 convolution is another 3x3 convolution.
    gen_hidden_1 = Conv2D(filters, kernel_size=width, activation='relu', padding='same')(sample_image)
    reconstruction = Conv2D(cifar_channels, kernel_size=width, activation='sigmoid', padding='same')(gen_hidden_1)

    # LIKELIHOOD
    # Note: in some models, pixel_std is not constant but is also an output of the model so that it
    # can indicate its own uncertainty.
    likelihood = DiagonalGaussianLikelihood(K.batch_flatten(reconstruction), pixel_std)

    # Combine the above parts into a single model
    if cls == 'vae':
        return VAE(inpt=inpt, latent=latent, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)
    elif cls == 'iwae':
        return IWAE(inpt=inpt, latent=latent, reconstruction=reconstruction, likelihood=likelihood, k_samples=k_samples)
