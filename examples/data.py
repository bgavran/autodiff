import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train, y_test = np_utils.to_categorical(y_train, 10), np_utils.to_categorical(y_test, 10)
print("Loaded data.")


def get_data(train, batch_size=32):
    if train:
        x, y = x_train, y_train
    else:
        x, y = x_test, y_test
    locations = np.random.randint(0, len(x), batch_size)
    return x[locations, ...], y[locations, ...]
