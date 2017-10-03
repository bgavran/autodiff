from utils import *

np.random.seed(1337)

x_val = np.random.rand(7)
x = Variable(x_val, name="x")

graph = Softmax(x, 2)
print(x())
print(graph())

graph = Grad(graph, x)

print(graph())

# graph = Grad(graph, x, expand_graph=True)
# print(graph())

plot_comp_graph(graph, view=False)
