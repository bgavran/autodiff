from utils import *

x1 = Variable(7, name="x1")
w1 = Variable(3, name="w1")

sqr = SquaredDifference(x1, w1)
print("Squared difference:", sqr())

graph = Grad(sqr, wrt=x1)
print("Derivative with respect to x1:", graph())

graph = Grad(graph, wrt=x1)
print("Second derivative with respect to x1:", graph())

plot_comp_graph(graph, view=True)
