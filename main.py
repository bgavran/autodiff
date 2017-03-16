from plot import *
from utils import *
from ops import *


class WeirdF(CompositeOperation):
    def graph(self):
        x0, x1 = self.children[0], self.children[1]
        w0, w1 = self.children[2], self.children[3]
        y = self.children[4]
        umn0 = w0
        umn1 = w1
        self.out = umn0 + umn1
        # self.out = w0 + w1


inp = Variable(name="x0"), Variable(name="x1"), \
      Variable(name="w0"), Variable(name="w1"), Variable(name="y")
with_respect_to = ["w0", "w1"]

graph = WeirdF(inp, "comp_graph")

m = GraphMeshgrid(with_respect_to)
grad = m.apply_to_function(graph.gradient_list, with_respect_to)
val = m.apply_to_function(graph.f)

p = Plotter()
p.plot_stream(m.w, grad, with_respect_to)
p.plot_value(m.w[0], m.w[1], val, with_respect_to)
plt.show(block=True)
