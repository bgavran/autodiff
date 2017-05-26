import numpy as np
from tqdm import tqdm


class GraphMeshgrid:
    epsilon = 0

    def __init__(self, x, w, y, func):
        self.x_vars = x
        self.w_vars = w
        self.w_names = [var.name for var in self.w_vars]
        self.y_var = y
        self.func = func
        self.x_len = len(self.x_vars)  # number (dimension) of input
        self.w_len = len(self.w_vars)

        self.xmax = 5
        self.wmax = 2
        self.p_exp = 5
        self.xn_points = 2 ** self.p_exp
        self.wn_points = 2 ** self.p_exp

        self.x = GraphMeshgrid.create_meshgrid(self.x_len, self.xmax, self.xn_points, GraphMeshgrid.epsilon)
        self.w = GraphMeshgrid.create_meshgrid(self.w_len, self.wmax, self.wn_points, 0)

        self.input_dict = {var.name: self.w[i] for i, var in enumerate(self.w_vars)}

    def apply_to_function(self, graph_function, *fun_args):
        """
        Iterates through every possible combination of inputs, calculates the output
        
        :param graph_function: 
        :param fun_args: 
        :return: 
        """
        l = list(np.nditer(self.x))
        res = []
        for x in tqdm(l):
            for i, var in enumerate(self.x_vars):
                self.input_dict[var.name] = x[i]
            self.input_dict[self.y_var.name] = self.func(x[0], x[1])
            res.append(graph_function(self.input_dict, *fun_args))

        return np.mean(res, axis=0)

    @staticmethod
    def create_meshgrid(lenn, maxx, points, offset):
        """
        When there is no x in the computational graph...?
        Why is the offset needed? So the mean of the x inputs is 1, which would, in the case of uniform distribution
        of x, make it equal as if its not there? As if instead of x*w there's only w?

        
        :param lenn: 
        :param maxx: 
        :param points: 
        :param offset: 
        :return: 
        """
        # add offsetting with x_d, y_d and z_d?
        # step = 2 * maxx / points
        # a = np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))
        return np.meshgrid(*(lenn * [np.linspace(-maxx, maxx, points) + offset]))
