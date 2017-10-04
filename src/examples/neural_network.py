from keras.datasets import mnist
from keras.utils import np_utils

from visualization.graph_visualization import *
from core.ops import *
from core.reshape import *

np.random.seed(1337)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train, y_test = np_utils.to_categorical(y_train, 10), np_utils.to_categorical(y_test, 10)
print("Loaded data.")


def get_data(batch_size=10):
    locations = np.random.randint(0, len(x_train), batch_size)
    return x_train[locations, ...], y_train[locations, ...]


def nn(x, w0):
    x_reshaped = Reshape(x, Shape(from_tuple=(-1, 28 * 28)))  # [batch_size, 28*28]
    graph = Sigmoid(x_reshaped @ w0)  # [batch_size, 10]
    return graph


def loss(nn_output, y):
    return SquaredDifference(nn_output, y)


w0 = Variable(np.random.randn(28 * 28, 10) * 0.01, name="w0")

for i in range(1000):

    x_val, y_val = get_data()
    x = Variable(x_val, name="x")
    y = Variable(y_val, name="y")

    network = nn(x, w0)

    loss_var = loss(network, y)

    grad = Grad(loss_var, w0)
    new_w0 = w0 - 0.01 * grad

    w0 = Variable(new_w0(), name="w0")

    if i % 100 == 0:
        print("step:", i)
        # print("network output:", network())
        # print("gradient:", grad())
        print("loss:", np.sum(loss_var()))
        print("-----------")

random_samples = np.random.randint(0, len(x_test), 10)
x, y = Variable(x_test[random_samples]), Variable(y_test[random_samples])

test_network = nn(x, w0)
test_loss = loss(test_network, y)

print("True y:", np.argmax(y_test[random_samples], -1))
print("Predicted y:", np.argmax(test_network(), -1))
print("Loss:", np.sum(test_loss()))
plot_comp_graph(new_w0, view=False)
