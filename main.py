from plot import *
from utils import *
from ops import *


class WeirdF(CompositeOperation):
    def graph(self):
        x0, x1 = self.children[0], self.children[1]
        w0, w1 = self.children[2], self.children[3]
        y = self.children[4]
        umn0 = w0
        umn1 = w1
        sigm0 = Sigmoid([umn0])
        sigm1 = Sigmoid([umn1])
        self.out = sigm0*sigm1


with_respect_to = ["w0", "w1"]
inp = Variable(name="x0"), Variable(name="x1"), \
      Variable(name="w0"), Variable(name="w1"), Variable(name="y")

graph = WeirdF(inp, "comp_graph")

###
mx = create_meshgrid(x_len, xmax, xn_points)
mw = create_meshgrid(len(with_respect_to), wmax, wn_points)

# rearanging the array, based on the wrt argument, should work for 3 dimensions also
myorder = [int(i[1]) for i in with_respect_to]
mw_input = [mw[i] for i in myorder]
###

dw_grad = np.zeros((len(with_respect_to), wn_points, wn_points))
grad = sum_function_outputs(graph.gradient_list, mx, mw_input, dw_grad, with_respect_to)

dw_val = np.zeros((wn_points, wn_points))
val = sum_function_outputs(graph.f, mx, mw_input, dw_val)

p = Plotter()
p.plot_stream(mw, grad, with_respect_to)
p.plot_value(mw_input[0], mw_input[1], val, with_respect_to)
plt.show(block=True)
