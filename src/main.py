from utils import *

x1 = Variable(7, name="x1")
w1 = Variable(3, name="w1")

sqrd = SquaredDifference(x1, w1)

grad1 = Grad(sqrd, x1)
grad2 = Grad(grad1, x1)


print("Squared difference:", sqrd())
print("Gradient with respect to x1:", grad1())
print("Gradient with respect to x1:", grad2())

plot_comp_graph(grad2, view=False)
