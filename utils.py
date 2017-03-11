import numpy as np
from tqdm import tqdm
from test_functions import *

p_exp = 5
x_len = 2  # number (dimension) of input
xmax = 5
xn_points = 2 ** p_exp

wmax = 2
wn_points = 2 ** p_exp


def create_meshgrid(lenn, maxx, points):
    # add offsetting with x_d, y_d and z_d?
    step = 2 * maxx / points
    return np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))


def gradient_meshgrid(graph_instance, wrt):
    w_len = len(wrt)
    mx = create_meshgrid(x_len, xmax, xn_points)
    mw = create_meshgrid(w_len, wmax, wn_points)

    dw = [np.zeros([wn_points for _ in range(w_len)]) for _ in range(w_len)]
    # rearanging the array, based on the wrt argument, should work for 3 dimensions also
    myorder = [int(i[1]) for i in wrt]
    mw_input = [mw[i] for i in myorder]
    # mw_input = mw

    l = list(np.nditer(mx))

    # iterating through every input, producing the output and summing the gradients
    for x in tqdm(l):

        y = if_func(x[0], x[1], 1)
        input_dict = {"x0": x[0], "x1": x[1], "w0": mw_input[0], "w1": mw_input[1], "y": y}

        graph_instance.compute_gradient(input_dict)
        for i, var in enumerate(wrt):
            dw[i] += graph_instance.gradient(wrt=var)

    # start point (uniform distribution), end point (calculated gradient)
    return mw, [i / mx[0].size for i in dw]
