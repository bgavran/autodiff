from core.ops import *
from utils import *

np.random.seed(1337)

x = Variable(name="x")
w = Variable(name="w")
w1 = Variable(name="w1")
y = Variable(name="y")

out = x
for i in range(5):
    out = out @ w
graph = out

idict = {"x": np.random.rand(2, 3), "w": np.random.rand(3, 3)}
print(graph.f(idict))
graph.compute_derivatives(idict)
print(graph.accumulate_all_gradients(wrt="w"))

plot_comp_graph(graph)


# meshgrids = GraphMeshgrid([x, x], [w, w1], y, test2)
# grad = meshgrids.apply_to_function(graph.accumulate_all_gradients_in_list, meshgrids.w_names)
# val = meshgrids.apply_to_function(graph.f)
#
# p = Plotter()
# p.plot_stream(meshgrids.w, grad, meshgrids.w_names)
# p.plot_value(meshgrids.w[0], meshgrids.w[1], val, meshgrids.w_names)
# plt.show(block=True)
