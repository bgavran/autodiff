from core.ops import *
from utils import *
import matplotlib.pyplot as plt

np.random.seed(1337)

x1 = Variable(name="x1")
w1 = Variable(name="w1")
graph = Tanh(x1)
graph = Grad(graph, wrt=x1, expand_when_graphed=True)
graph = Grad(graph, wrt=x1, expand_when_graphed=True)

plot_comp_graph(graph, view=False)
inpd = {x1: 10, w1: 15}
res = graph(inpd)
print(res)

# x1 = Variable(name="x1")
#
# input_dict = {x1: np.arange(-7, 7, 0.01, dtype=np.float32)}
# graph = Tanh(x1)
# deriv_1 = Grad(graph, wrt=x1)
# deriv_2 = Grad(deriv_1, wrt=x1)
#
# plot_comp_graph(deriv_2, view=True)
# plt.plot(x1(input_dict), graph(input_dict),
#          x1(input_dict), deriv_1(input_dict),
#          x1(input_dict), deriv_2(input_dict))
#
# plt.show()


# meshgrids = GraphMeshgrid([x, x], [w, w1], y, test2)
# grad = meshgrids.apply_to_function(graph.accumulate_all_gradients_in_list, meshgrids.w_names)
# val = meshgrids.apply_to_function(graph.f)
#
# p = Plotter()
# p.plot_stream(meshgrids.w, grad, meshgrids.w_names)
# p.plot_value(meshgrids.w[0], meshgrids.w[1], val, meshgrids.w_names)
# plt.show(block=True)
