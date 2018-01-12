from data.my_mnist import img_rows, img_cols
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K


def render_grid(sample_var, reconstruction_var):
    # Visualize model by running the P-model forward from a grid of points in the latent space
    # ranging from -3 to +3.
    num_grid = 15
    grid_pts = np.linspace(-3, 3, num_grid)

    # Create a keras function that takes latent (x, y) as input and produces images as output
    renderer = K.function([sample_var], [reconstruction_var])
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
