import numpy as np
from tqdm import tqdm
from test_functions import *


class GraphMeshgrid:
    epsilon = 1

    def __init__(self, with_respect_to):
        self.p_exp = 5

        self.x_len = 2  # number (dimension) of input
        self.xmax = 5
        self.wmax = 2
        self.xn_points = 2 ** self.p_exp
        self.wn_points = 2 ** self.p_exp

        self.x = GraphMeshgrid.create_meshgrid(self.x_len, self.xmax, self.xn_points, GraphMeshgrid.epsilon)
        self.w = GraphMeshgrid.create_meshgrid(len(with_respect_to), self.wmax, self.wn_points, 0)

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
    def create_meshgrid(lenn, maxx, points, offset):
        # Why is the offset needed? So the mean of the x inputs is 1, whic would, in the case of uniform distribution of
        # x, make it equal as if its not there? As if instead of x*w there's only w?

        # add offsetting with x_d, y_d and z_d?
        # step = 2 * maxx / points
        # a = np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))
        return np.meshgrid(*(lenn * [np.linspace(-maxx, maxx, points) + offset]))
