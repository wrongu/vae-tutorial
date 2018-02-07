from keras.datasets import cifar10


img_rows, img_cols, img_channels = 32, 32, 3
img_pixels = img_rows * img_cols * img_channels

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert uint8 [0-255] to float32 [0.0-1.0]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
