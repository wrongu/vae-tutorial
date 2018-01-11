from models import gaussian_mnist, fit_vae
from data.my_mnist import x_train, x_test, img_rows, img_cols
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

# PART 1: CREATE AND TRAIN MODEL

vae = gaussian_mnist(latent_dim=2, pixel_std=.05)
fit_vae(vae, x_train, x_test, epochs=100, weights_file='01-weights.h5')

# PART 2: VISUALIZE RESULTS

# Visualize results by running the P-model forward from a grid of points in the latent space
# ranging from -3 to +3.
num_grid = 15
grid_pts = np.linspace(-3, 3, num_grid)

# Create a keras function handle that takes latent points as input and produces images as output
renderer = K.function([vae.latent.sample], [vae.reconstruction])
latent_input = np.zeros((1, 2))  # must have shape (1, 2) not (2,)

# Allocate space for the final (num_grid * rows, num_grid * cols) image
final_image = np.zeros((num_grid * img_rows, num_grid * img_cols))

# Populate each sub-image one at a time
for i in range(num_grid):
    for j in range(num_grid):
        latent_input[:] = (grid_pts[i], grid_pts[j])
        img = renderer([latent_input])[0]
        final_image[j * img_rows:(j + 1) * img_rows, i * img_cols:(i + 1) * img_cols] = \
            np.reshape(img, (img_rows, img_cols))

plt.imshow(final_image, cmap='gray', extent=(-3, 3, -3, 3))
plt.show()
