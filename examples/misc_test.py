import numpy as np
import automatic_differentiation as ad

a = ad.Variable(3, name="a")
b = ad.Variable(4, name="b")

c = a * b

x = ad.Sigmoid(ad.grad(c, [a])[0])
y = ad.grad(ad.Sigmoid(c), [a])[0]

print(x())
print(y())
t = 1
