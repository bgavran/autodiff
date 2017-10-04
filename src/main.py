from visualization.graph_visualization import plot_comp_graph
from core.ops import *

np.random.seed(1337)

x_val = np.random.randn(1, 10)
x = Variable(x_val, name="x")

print(x())

graph = Softmax(x)
print("------------")
print(graph())

graph = Grad(graph, x)
print("------------")
print(graph())

plot_comp_graph(graph, view=False)
