from plot import *
from utils import *
from ops import *
from test_functions import *


class WeirdF(CompositeOperation):
    def __init__(self, children, name=""):
        # Needs to be initialized this way because this composite graph is at the bottom (leaf nodes) and data is
        # coming in through here. Maybe create a new class that handles this?
        self.x0, self.x1 = Variable(children[0]), Variable(children[1])
        self.w0, self.w1 = Variable(children[2], name="w0"), Variable(children[3], name="w1")
        chldrn = self.x0, self.x1, self.w0, self.w1
        super().__init__(chldrn, name)
        self.out = None
        self.graph()

    def graph(self):
        y = Variable(if_func(self.children[0](), self.children[1](), 1))
        umn1 = self.x0 * self.w0
        umn2 = self.x1 * self.w1
        sigm = Sigmoid([umn1])
        umn = sigm * umn2
        self.out = SquareCost([umn, y])


with_respect_to = ["w0", "w1"]

grad = gradient_meshgrid(WeirdF, with_respect_to)
Plotter().plot_stream(*grad, with_respect_to)

input()
