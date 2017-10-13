import numpy as np
import automatic_differentiation as ad

np.random.seed(1337)

w0_val = np.random.rand(2, 3)
w1_val = np.random.rand(2, 3)

w0 = ad.Variable(w0_val, name="w")
w1 = ad.Variable(w1_val, name="w")

graph = ad.SoftmaxCEWithLogits(labels=w0, logits=w1)

print(graph())

graph.plot_comp_graph()
