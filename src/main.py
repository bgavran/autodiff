from utils import *

x_val = np.random.randn(2, 3, 5)

x = Variable(x_val, name="x1")

es = EinSum("ijk->ij", x)

grad = Grad(es, x)

print(x().shape)
print(grad().shape)

