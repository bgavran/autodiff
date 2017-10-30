import numpy as np
import automatic_differentiation as ad

x = ad.Variable(np.random.randn(5, 3), name="x")
b = ad.Variable(np.random.randn(5, 1), name="b")

c = ad.Concat(x, b, axis=1)

print("c ==", c().shape, "\n")

graph = ad.grad(c, [x])[0]
print(graph().shape)
