from models import gaussian_cnn_cifar10, fit_vae
from data.my_cifar10 import x_train, x_test, img_rows, img_cols, img_channels
from visualize import render_grid

# Create and train the model
vae = gaussian_cnn_cifar10('vae', latent_dim=32, pixel_std=.05, k=4, filters=32)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='weights.h5')

# Visualize results
render_grid(vae.latent.flat_samples, vae.reconstruction, ndim=32, img_rows=img_rows,
            img_cols=img_cols, img_channels=img_channels)
