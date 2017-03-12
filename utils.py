import numpy as np
from tqdm import tqdm
from test_functions import *

p_exp = 5
x_len = 2  # number (dimension) of input
xmax = 5
xn_points = 2 ** p_exp

wmax = 5
wn_points = 2 ** p_exp


def create_meshgrid(lenn, maxx, points):
    # add offsetting with x_d, y_d and z_d?
    step = 2 * maxx / points
    return np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))


def sum_function_outputs(graph_fun, mx, mw, dw, *fun_args):
    # iterating through every input, producing the output and summing the gradients
    l = list(np.nditer(mx))
    for x in tqdm(l):
        y = if_func(x[0], x[1], 1)
        input_dict = {"x0": x[0], "x1": x[1], "w0": mw[0], "w1": mw[1], "y": y}

        dw += graph_fun(input_dict, *fun_args)

    # start point (uniform distribution), end point (calculated gradient)
    return dw / mx[0].size
