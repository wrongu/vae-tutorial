from models import gaussian_cnn_cifar10, fit_vae
from data.my_cifar10 import x_train, x_test
from visualize import render_grid

# Create and train the model
vae = gaussian_cnn_cifar10('vae', latent_dim=16, pixel_std=.05, k=4, filters=16)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='weights.h5')

# Visualize results
render_grid(vae.latent.flat_samples, vae.reconstruction)
