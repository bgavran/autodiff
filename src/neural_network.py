from utils import *

# from keras.datasets import mnist

np.random.seed(1337)


# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# flatten the data?

@checkpoint
def nn(x, w0):
    print("steppp")

    x_reshaped = Reshape(x, Shape(from_tuple=(-1, 28 * 28)))  # [batch_size, 28*28]
    graph = Softmax(x_reshaped @ w0)  # [batch_size, 10]
    return graph


@checkpoint
def optimizer(var, var_grad):
    alpha = 0.1
    return var - alpha * var_grad


@checkpoint
def update_weight(out, w):
    w_grad = Grad(out, wrt=w, expand_graph=True)
    return optimizer(w, w_grad)


def get_data():
    batch_size = 3
    return np.random.randn(batch_size, 28, 28), np.random.randint(batch_size, 10)


w0 = Variable(np.random.randn(28 * 28, 10), name="w0")  # [batch_size, 28, 28]
w1 = Variable(np.random.randn(5, 7), name="w1")  # [batch_size, 28*28, 10]

n_steps = 1000
for i in range(n_steps):
    print("step:", i)

    x_val, y_val = get_data()
    x = Variable(x_val, name="x")
    y = Variable(y_val, name="y")

    network = nn(x, w0)

    w0_new = update_weight(network, w0)
    w0.value = w0_new()

    nn_val = network()
    print("nn_sum:", np.sum(nn_val))
    print("-----------")
plot_comp_graph(w0_new, view=False)
