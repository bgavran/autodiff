from utils import *
import matplotlib.pyplot as plt

np.random.seed(1337)

x1 = Variable(name="x1")
w1 = Variable(name="w1")

# graph = SquaredDifference(x1, x1)
graph = ReLU(x1)
graph = Grad(graph, wrt=x1, expand_when_graphed=True)
# graph = Grad(graph, wrt=x1, expand_when_graphed=True)
# graph = Grad(graph, wrt=x1, expand_when_graphed=True)
# graph = Grad(graph, wrt=x1, expand_when_graphed=True)

plot_comp_graph(graph, view=False)
inpd = {x1: 10, w1: 6}
# # inpd = {x1: np.arange(-5, 5, 0.1), w1: 15}
res = graph(inpd)
print(res)

# x1 = Variable(name="x1")
#
# my_input_dict = {x1: np.arange(-7, 7, 0.01, dtype=np.float32)}
# graph = Tanh(x1)
# deriv_1 = Grad(graph, wrt=x1)
# deriv_2 = Grad(deriv_1, wrt=x1)
#
# plot_comp_graph(deriv_2, view=True)
# plt.plot(x1(my_input_dict), graph(my_input_dict),
#          x1(my_input_dict), deriv_1(my_input_dict),
#          x1(my_input_dict), deriv_2(my_input_dict))
#
# plt.show()
