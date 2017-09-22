from utils import *

x1 = Variable(name="x1")
w1 = Variable(name="w1")

sqr = SquaredDifference(x1, w1, graph_expand=False)

grad1 = Grad(sqr, wrt=x1, graph_expand=False)
grad2 = Grad(grad1, wrt=x1, graph_expand=False)

inpd = {x1: 7, w1: 3}

print("Squared difference:", sqr(inpd))
print("Derivative with respect to x1:", grad1(inpd))
print("Second derivative with respect to x1:", grad2(inpd))

plot_comp_graph(grad2, view=False)
