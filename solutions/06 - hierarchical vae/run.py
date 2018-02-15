from models import gaussian_mnist_hvae, fit_vae
from data.my_mnist import x_train, x_test
from visualize import render_grid
import matplotlib.pyplot as plt
import numpy as np

# Create and train the model
vae = gaussian_mnist_hvae('vae', latent_dim=16, pixel_std=.05, k=4)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='weights.h5')

# Visualize results for variability in each layer given a set value in other layers
nlayers = len(vae.latents)
presets = [np.random.randn(1, l.dim).astype(np.float32) for l in vae.latents]
for i in range(nlayers):
    plt.subplot(1, nlayers, i + 1)
    other_latents = [l.flat_samples for (j, l) in enumerate(vae.latents) if j != i]
    other_values = [pre for (j, pre) in enumerate(presets) if j != i]
    render_grid(vae.latents[i].flat_samples, vae.reconstruction, ndim=16,
                preset_vars=other_latents, preset_values=other_values, display=False)

plt.show()
