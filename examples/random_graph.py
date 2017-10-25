import automatic_differentiation as ad

w0 = ad.Variable(2, name="w0")
w1 = ad.Variable(3, name="w1")

graph = w0 * w1
print(graph())
w0_grad = ad.grad(graph, [w0])[0]
print(w0_grad())
