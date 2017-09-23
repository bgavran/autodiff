from utils import *

np.random.seed(1337)

# Difference between a checkpoint and a composite operation?
# Similar question: difference between eval() and graph() ?
# There should be some connection?

"""
TODO list:
* Perhaps I should first try making the network work and then see what needs to be done?

* Restructuring code (is Grad a class, just a function and how do checkpoints work with it?)
* If there are N weight variables that need to be updated, how are computational graphs created then?
* Computational graph expand_graphed doesn't seem to be working when there's two grads in a row? Might 
get automatically solved when checkpoint and grad_fn gets solved?

"""


def get_data():
    # Returns x1, w0_val and w1_val, randomly generated
    # How should it work exactly?
    return np.random.randn(2, 3), np.random.randn(3, 5), np.random.randn(5, 7)


@composite_wrapper
def nn(inp, w0):
    graph = Sigmoid(inp @ w0, expand_graph=False)
    return graph


@composite_wrapper
def optimizer(var, var_grad):
    alpha = 0.1
    return var - alpha * var_grad


@checkpoint
def step(x1, w0):
    print("STEPPPPPPPPPPPPPPPPP")
    network = nn(x1, w0, expand_graph=True)

    w0_grad = Grad(network, wrt=w0, expand_graph=True)
    w0 = optimizer(w0, w0_grad)
    return w0


x1_val = np.random.randn(2, 3)
w0_val = np.random.randn(3, 5)

x1 = Variable(x1_val, name="x1")
w0 = Variable(w0_val, name="w0_val")

n_steps = 10
for i in range(n_steps):
    if i % 10 == 0:
        net_output = nn(x1, w0)()
        print("step", i, "network sum:", np.sum(net_output))
    w0 = step(x1, w0)

plot_comp_graph(w0, view=False)
