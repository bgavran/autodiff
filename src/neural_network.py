from utils import *

np.random.seed(1337)

# Difference between a checkpoint and a composite operation?
# Similar question: difference between eval() and graph() ?
# There should be some connection?

# The difference is that eval() needs a feed_dict, while graph() doesn't?
# Edit: eval doesn't need a feed_dict anymore? :)

"""
TODO list:
* Perhaps I should first try making the network work and then see what needs to be done?

* Adding checkpoints
* Restructuring code (is Grad a class, just a function and how do checkpoints work with it?)
* Removing feed dict and enabling another way of inputting data into the network
* If there are N weight variables that need to be updated, how are computational graphs created then?

"""


def get_data():
    # Returns x1, w0_val and w1_val, randomly generated
    # How should it work exactly?
    return np.random.randn(2, 3), np.random.randn(3, 5), np.random.randn(5, 7)


@CompositeWrapper.from_function
def nn(inp, w0):
    graph = Sigmoid(inp @ w0, graph_expand=False)
    return graph


@CompositeWrapper.from_function
def optimizer(var, var_grad):
    alpha = 0.1
    return var - alpha * var_grad


@CompositeWrapper.from_function
def step(x1, w0):
    network = nn(x1, w0, graph_expand=True)

    w0_grad = Grad(network, wrt=w0, graph_expand=True)
    w0 = optimizer(w0, w0_grad, graph_expand=True)
    return w0


x1_val = np.random.randn(2, 3)
w0_val = np.random.randn(3, 5)

x1 = Variable(x1_val, name="x1")
w0 = Variable(w0_val, name="w0_val")

n_steps = 5
for i in range(n_steps):
    print("step", i)
    print("network sum:", np.sum(nn(x1, w0)()))
    w0 = step(x1, w0, graph_expand=False)

plot_comp_graph(w0, view=False)

#
# print(inpd[w0_val])
# res = w0_val[]
