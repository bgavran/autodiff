import numpy as np
from tqdm import tqdm

"""

# Old visualization code:

meshgrids = GraphMeshgrid([x_val, x_val], [w_val, w1_val], y, test2)
grad = meshgrids.apply_to_function(graph.accumulate_all_gradients_in_list, meshgrids.w_names)
val = meshgrids.apply_to_function(graph.f)

p = Plotter()
p.plot_stream(meshgrids.w_val, grad, meshgrids.w_names)
p.plot_value(meshgrids.w_val[0], meshgrids.w_val[1], val, meshgrids.w_names)
plt.show(block=True)


"""


class GraphMeshgrid:
    epsilon = 0

    def __init__(self, x_vars, w_vars, y, func):
        # TODO write  documentation
        # w_var needs to be two dimensional
        self.x_vars = x_vars
        self.w_vars = w_vars
        self.w_names = ["w0_val", "w1_val"]
        self.y_var = y
        self.func = func
        self.x_len = len(self.x_vars)  # number (dimension) of input
        self.w_len = 2

        self.xmax = 5
        self.wmax = 2
        self.p_exp = 5
        self.xn_points = 2 ** self.p_exp
        self.wn_points = 2 ** self.p_exp

        self.x = GraphMeshgrid.create_meshgrid(self.x_len, self.xmax, self.xn_points, GraphMeshgrid.epsilon)
        self.w = GraphMeshgrid.create_meshgrid(self.w_len, self.wmax, self.wn_points, 0)

        self.input_dict = {self.w_names[i]: self.w[i] for i in range(self.w_len)}

    def apply_to_function(self, graph_function, *fun_args):
        """
        Iterates through every possible combination of inputs, calculates the output
        
        :param graph_function: 
        :param fun_args: 
        :return: 
        """
        x_list = list(np.nditer(self.x))
        res = []
        for x in tqdm(x_list):
            for i, var in enumerate(self.x_vars):
                self.input_dict[var.name] = x[i]
            self.input_dict[self.y_var.name] = self.func(x[0], x[1])
            res.append(graph_function(self.input_dict, *fun_args))

        return np.mean(res, axis=0)

    @staticmethod
    def create_meshgrid(lenn, maxx, points, offset):
        """
        When there is no x_val in the computational graph...?
        Why is the offset needed? So the mean of the x_val inputs is 1, which would, in the case of uniform distribution
        of x_val, make it equal as if its not there? As if instead of x_val*w_val there's only w_val?

        
        :param lenn: 
        :param maxx: 
        :param points: 
        :param offset: 
        :return: 
        """
        # add offsetting with x_d, y_d and z_d?
        # step = 2 * maxx / points
        # true = np.meshgrid(*(lenn * [np.arange(-maxx, maxx, step)]))
        return np.meshgrid(*(lenn * [np.linspace(-maxx, maxx, points) + offset]))
