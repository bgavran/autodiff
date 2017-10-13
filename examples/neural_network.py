import numpy as np
import automatic_differentiation as ad

from examples.data import get_data


@ad.module_wrapper
def nn(x, w0):
    x_reshaped = ad.Reshape(x, ad.Shape(from_tuple=(-1, 28 * 28)))  # [batch_size, 28*28]
    # TODO Add working softmax here!
    graph = ad.Softmax(x_reshaped @ w0)  # [batch_size, 10]
    return graph


@ad.module_wrapper
def loss(nn_output, y):
    return ad.SquaredDifference(nn_output, y)


@ad.module_wrapper
def optimizer(w, w_grad):
    lr = 0.001
    return w - lr * w_grad


w0 = ad.Variable(np.random.randn(28 * 28, 10) * 0.01, name="w0")

for i in range(1000):
    x_val, y_val = get_data(train=True)
    x, y = ad.Variable(x_val, name="x"), ad.Variable(y_val, name="y")

    network = nn(x, w0)
    loss_var = loss(network, y)

    w0_grad = ad.Grad(loss_var, w0)
    new_w0 = optimizer(w0, w0_grad)

    w0 = ad.Variable(new_w0(), name="w0")

    if i % 100 == 0:
        print("Step:", i)
        print("Loss:", np.sum(loss_var()))
        print("-----------")

print("Testing...")
x, y = get_data(train=False)
x, y = ad.Variable(x, name="x"), ad.Variable(y, name="y")

test_network = nn(x, w0)
test_loss = loss(test_network, y)

true = np.argmax(y(), -1)
pred = np.argmax(test_network(), -1)
print("True y:", true)
print("Predicted y:", pred)
print("Last batch accuracy:", np.mean(true == pred))
print("Loss:", np.sum(test_loss()))

new_w0.plot_comp_graph(view=True)
