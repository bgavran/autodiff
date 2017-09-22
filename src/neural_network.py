from utils import *

np.random.seed(1337)

# Difference between a checkpoint and a composite operation?
# Similar question: difference between eval() and graph() ?
# There should be some connection?

# The difference is that eval() needs a feed_dict, while graph() doesn't?

"""
TODO list:
* Perhaps I should first try making the network work and then see what needs to be done?

* Adding checkpoints
* Restructuring code (is Grad a class, just a function and how do checkpoints work with it?)
* Removing feed dict and enabling another way of inputting data into the network
* If there are N weight variables that need to be updated, how are computational graphs created then?

"""


def get_data():
    # Returns x1, w0 and w1, randomly generated
    # How should it work exactly?
    return np.random.randn(2, 3), np.random.randn(3, 5), np.random.randn(5, 7)


@CompositeWrapper.from_function
def nn(inp, w0):
    graph = Sigmoid(inp @ w0, graph_expand=False)
    return graph

@CompositeWrapper.from_function
def optimizer(var, var_grad):
    alpha = 0.001
    return var - alpha * var_grad


x1 = Variable(name="x1")
w0 = Variable(name="w0")


@CompositeWrapper.from_function
def step(x1, w0):
    network = nn(x1, w0, graph_expand=False)

    w0_grad = Grad(network, wrt=w0, graph_expand=False)
    w0 = optimizer(w0, w0_grad, graph_expand=False)
    return w0


n_steps = 3
for i in range(n_steps):
    print("step", i)
    w0 = step(x1, w0, graph_expand=False)

print("asddf")
plot_comp_graph(w0, view=False)

inpd = {x1: np.random.randn(2, 3), w0: np.random.randn(3, 5)}

print(inpd[w0])
res = w0[]

