import numpy as np
from tqdm import tqdm
from test_functions import *


class GraphMeshgrid:
    def __init__(self, with_respect_to):
        self.p_exp = 5

        self.x_len = 2  # number (dimension) of input
        self.xmax = 5
        self.wmax = 2
        self.xn_points = 2 ** self.p_exp
        self.wn_points = 2 ** self.p_exp

        self.x = GraphMeshgrid.create_meshgrid(self.x_len, self.xmax, self.xn_points)
        self.w = GraphMeshgrid.create_meshgrid(len(with_respect_to), self.wmax, self.wn_points)

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

    @staticmethod
    def create_meshgrid(lenn, maxx, points):
        # add offsetting with x_d, y_d and z_d?
        # step = 2 * maxx / points
        # a = np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))
        epsilon = 1
        return np.meshgrid(*(lenn * [np.linspace(-maxx, maxx, points) + epsilon]))
