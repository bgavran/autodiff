import numpy as np
import automatic_differentiation as ad

np.random.seed(1337)

x = ad.Variable(7, name="x")
w = ad.Variable(3, name="w")

print(x())

graph = 2 * x + w
print("------------")
print(graph())

graph = ad.Grad(graph, x, expand_graph=True)
print("------------")
print(graph())

ad.plot_comp_graph(graph, view=True)
