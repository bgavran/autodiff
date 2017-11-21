import numpy as np
import matplotlib.pyplot as plt
import autodiff as ad

x = ad.Variable(np.linspace(-7, 7, 200), name="x")

fn = ad.Tanh


def diff_n_times(graph, wrt, n):
    for _ in range(n):
        graph = ad.grad(graph, [wrt])[0]
    return graph


plt.plot(x(), diff_n_times(fn(x), x, 0)(),
         x(), diff_n_times(fn(x), x, 1)(),
         x(), diff_n_times(fn(x), x, 2)(),
         x(), diff_n_times(fn(x), x, 3)(),
         x(), diff_n_times(fn(x), x, 4)())

plt.title("Visualization of higher order derivatives of tanh")
plt.show()
# ad.grad(fn(x), [x])[0].plot_comp_graph()
