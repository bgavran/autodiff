from plot import *
from utils import *
from ops import *


class WeirdF(CompositeOperation):
    def graph(self):
        x0, x1 = self.children[0], self.children[1]
        w0, w1 = self.children[2], self.children[3]
        y = self.children[4]
        umn0 = x0 * w0
        umn1 = x1 * w1
        sigm = Sigmoid([umn0])
        umn = sigm * umn1
        p = SquareCost([umn, y])
        self.out = p


with_respect_to = ["w0", "w1"]
inp = Variable(name="x0"), Variable(name="x1"), \
      Variable(name="w0"), Variable(name="w1"), Variable(name="y")

graph = WeirdF(inp, "comp_graph")
grad = gradient_meshgrid(graph, with_respect_to)

Plotter().plot_stream(*grad, with_respect_to)

plt.show(block=True)
