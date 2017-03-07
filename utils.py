import numpy as np
from tqdm import tqdm

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


def gradient_meshgrid(graph, wrt):
    w_len = len(wrt)
    mx = create_meshgrid(x_len, xmax, xn_points)
    mw = create_meshgrid(w_len, wmax, wn_points)

    dw = [np.zeros([wn_points for _ in range(w_len)]) for _ in range(w_len)]
    # rearanging the array, based on the wrt argument, should work for 3 dimensions also
    myorder = [int(i[1]) for i in wrt]
    mw_input = [mw[i] for i in myorder]

    # def compute_grad(x, wrt):
    #     graph = function(x, mw_input, param)
    #     graph.compute_gradient()
    #     return [graph.gradient(var=var) for var in wrt]

    l = list(np.nditer(mx))

    # iterating through every input, producing the output and summing the gradients
    for x in tqdm(l):
        input_ = *x, *mw_input
        graph_instance = graph(input_, "comp_graph")
        graph_instance.compute_gradient()
        for i, var in enumerate(wrt):
            dw[i] += graph_instance.gradient(wrt=var)

    # start point (uniform distribution), end point (calculated gradient)
    return mw, [i / mx[0].size for i in dw]
