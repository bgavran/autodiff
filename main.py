from value_plot import *
from meshgrid import *
from ops import *
from test_functions import *
from utils import *

x0, x1 = Variable(name="x0"), Variable(name="x1")
x2 = Variable(name="x2")
y = Variable(name="y")
w0, w1 = Variable(name="w0"), Variable(name="w1")
w2, w3 = Variable(name="w2"), Variable(name="w3")
w4, w5 = Variable(name="w2"), Variable(name="w3")

umn0 = x0 * w0 * w0
umn1 = x1 * w1 * w1

layer1 = DenseLayer([x0, x1, x2])
# umn2 = x0 * w2
# umn3 = x1 * w3

# sum0 = umn0 + umn1 * layer1
sum0 = layer1*2
# sum1 = umn2 + umn3

# umn4 = sum0 * w4
# umn5 = sum1 * w5

# sum_last = umn4 + umn5

nonlin = Sigmoid(sum0)

cost = SquaredDifference(nonlin, y)

graph = cost + 3
plot_comp_graph(graph)

# meshgrids = GraphMeshgrid([x0, x1], [w1, w0], y, if_func)
# grad = meshgrids.apply_to_function(graph.gradient_list, meshgrids.w_names)
# val = meshgrids.apply_to_function(graph.f)
#
# p = Plotter()
# p.plot_stream(meshgrids.w, grad, meshgrids.w_names)
# p.plot_value(meshgrids.w[0], meshgrids.w[1], val, meshgrids.w_names)
# plt.show(block=True)
