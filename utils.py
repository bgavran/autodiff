import numpy as np
from tqdm import tqdm
from test_functions import *

p_exp = 5
x_len = 2  # number (dimension) of input
xmax = 5
xn_points = 2 ** p_exp

wmax = 2
wn_points = 2 ** p_exp


class GraphMeshgrid:
    def __init__(self, with_respect_to):
        self.x = create_meshgrid(x_len, xmax, xn_points)
        self.w = create_meshgrid(len(with_respect_to), wmax, wn_points)

        # rearanging the array, based on the wrt argument, should work for 3 dimensions also
        myorder = [int(i[1]) for i in with_respect_to]
        # ow = ordered weight
        self.ow = [self.w[i] for i in myorder]

    def apply_to_function(self, graph_function, *fun_args):
        # iterating through every input, producing the output and summing the gradients
        l = list(np.nditer(self.x))
        all_x = np.array(
            [graph_function({"x0": x[0], "x1": x[1], "w0": self.ow[0], "w1": self.ow[1], "y": if_func(x[0], x[1], 1)},
                            *fun_args)
             for x in
             tqdm(l)])

        return np.sum(all_x, axis=0) / len(l)


def create_meshgrid(lenn, maxx, points):
    # add offsetting with x_d, y_d and z_d?
    step = 2 * maxx / points
    return np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))
