from models import gaussian_mnist, fit_vae
from data.my_mnist import x_train, x_test
from visualize import render_grid

# Create and train the VAE model
vae = gaussian_mnist('vae', latent_dim=2, pixel_std=.05, k=8)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='vae-weights.h5')

# Visualize results
render_grid(vae.latent.flat_samples, vae.reconstruction)

# Create and train the IWAE model
vae = gaussian_mnist('iwae', latent_dim=2, pixel_std=.05, k=8)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='iwae-weights.h5')

# Visualize results
render_grid(vae.latent.flat_samples, vae.reconstruction)
