from plot import *
from utils import *
from ops import *
from test_functions import *

x0, x1, w0, w1, y = Variable(name="x0"), Variable(name="x1"), \
                    Variable(), Variable(name="w1"), Variable(name="y")

umn0 = x0 * w0
umn1 = x1 * w1
sigm = Sigmoid(umn0) * x0
umn = sigm * umn1
cost = SquareDiff(umn, y)

graph = cost

meshgrids = GraphMeshgrid([x0, x1], [w1, w0], y, if_func)
grad = meshgrids.apply_to_function(graph.gradient_list, meshgrids.w_names)
val = meshgrids.apply_to_function(graph.f)

p = Plotter()
p.plot_stream(meshgrids.w, grad, meshgrids.w_names)
p.plot_value(meshgrids.w[0], meshgrids.w[1], val, meshgrids.w_names)
plt.show(block=True)
