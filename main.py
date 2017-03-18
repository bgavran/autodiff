from plot import *
from utils import *
from ops import *

x0, x1, w0, w1, y = Variable(name="x0"), Variable(name="x1"), \
                    Variable(name="w0"), Variable(name="w1"), Variable(name="y")

umn0 = w0
umn1 = w1
graph = Gauss(w0) + Sigmoid(3 * w1)

with_respect_to = ["w0", "w1"]

meshgrids = GraphMeshgrid(with_respect_to)
grad = meshgrids.apply_to_function(graph.gradient_list, with_respect_to)
val = meshgrids.apply_to_function(graph.f)

p = Plotter()
p.plot_stream(meshgrids.w, grad, with_respect_to)
p.plot_value(meshgrids.w[0], meshgrids.w[1], val, with_respect_to)
plt.show(block=True)
