from utils import *

x1 = Variable(7, name="x1")
w1 = Variable(3, name="w1_val")

sqr = SquaredDifference(x1, w1, graph_expand=False)

grad1 = Grad(sqr, wrt=x1, graph_expand=False)
grad2 = Grad(grad1, wrt=x1, graph_expand=False)


print("Squared difference:", sqr())
print("Derivative with respect to x1:", grad1())
print("Second derivative with respect to x1:", grad2())

plot_comp_graph(grad2, view=False)
