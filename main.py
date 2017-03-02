from plot import *
from utils import *
from ops import *
from test_functions import *


def graph(x, w, param=0):
    """

    :param x: Data; Must be a scalar?
    :param w: Parameters; Can be a scalar or a tensor
    :return: Differentiable operation which is performed on data and parameters
    """
    # x_var = [Variable(x_i, name="x" + str(i)) for (i, x_i) in enumerate(x)]
    # w_var = [Variable(w_i, name="w" + str(i)) for (i, w_i) in enumerate(w)]

    y = Variable(if_func(x[0], x[1], param))

    x0, x1 = Variable(x[0]), Variable(x[1])
    w0, w1 = Variable(w[0], name="w0"), Variable(w[1], name="w1")

    umn1 = x0 * w0
    umn2 = x1 * w1

    sigm = Sigmoid([umn1])

    umn = sigm * umn2

    res = SquareCost([umn, y])

    return res


wrt = ["w0", "w1"]

grad = gradient_meshgrid(graph, wrt, [1])
Plotter().plot_stream(*grad, wrt)

input()
