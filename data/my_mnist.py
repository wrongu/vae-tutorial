from keras.datasets import mnist


img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten data to (num-images, pixels)
x_train = x_train.reshape(x_train.shape[0], img_pixels)
x_test = x_test.reshape(x_test.shape[0], img_pixels)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
