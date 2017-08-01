from core.ops import *
from utils import *

np.random.seed(1337)

# class DenseLayer(CompositeOperation):
#     def __init__(self, inputs, weights, activation=Sigmoid, name="DenseLayer"):
#         super().__init__([inputs, weights], name)
#         self.inputs = self.children[0]
#         self.weights = self.children[1]
#         self.activation = activation
#         self.graph()
#
#     def graph(self):
#         self.out = self.activation(self.inputs @ self.weights)
#
#
# class Optimizer(CompositeOperation):
#     def __init__(self, weight_gradients, alpha=0.0001, name="Optimizer"):
#         super().__init__([weight_gradients], name)
#         self.weight_gradients = weight_gradients
#         self.alpha = alpha
#         self.graph()
#
#     def graph(self):
#         self.out = self.weight_gradients * self.alpha
#
#
# class Step(CompositeOperation):
#     def __init__(self, inputs, weights, weight_gradients, nn, optimizer, name="Step"):
#         super().__init__([inputs, weights, weight_gradients], name)
#         self.inputs = inputs
#         self.weights = weights
#         self.weight_gradients = weight_gradients
#         self.nn = nn
#         self.optimizer = optimizer
#         self.graph()
#
#     def graph(self):
#         new_weights = self.weights - self.optimizer(self.weight_gradients)
#         new_output = self.nn(self.inputs, new_weights)
#         self.out = [new_output, new_weights]
#
#
# w = Variable(name="w")
# w_initial_grads = Variable(name="w_grads")
#
# out_sum = 0
# w_grads = w_initial_grads
# for i in range(5):
#     x = Variable(name="x" + str(i))
#
#     w_grads = Variable(name="w" + str(i) + "_grads")
#     s = Step(x, w, w_grads, DenseLayer, Optimizer)
#     out, w = s.out
#     out_sum += out

x = Variable(name="x")
w = Variable(name="w")

output = x
for i in range(5):
    output @= w

graph = Grad(output)

plot_comp_graph(graph, show_all_ops=True)

input_dict = {"x": np.random.rand(3, 5), "w": np.random.rand(5, 5)}

print(graph.f(input_dict))
# graph.compute_derivatives(input_dict)
# print(graph.accumulate_all_gradients(wrt="x"))

# meshgrids = GraphMeshgrid([x, x], [w, w1], y, test2)
# grad = meshgrids.apply_to_function(graph.accumulate_all_gradients_in_list, meshgrids.w_names)
# val = meshgrids.apply_to_function(graph.f)
#
# p = Plotter()
# p.plot_stream(meshgrids.w, grad, meshgrids.w_names)
# p.plot_value(meshgrids.w[0], meshgrids.w[1], val, meshgrids.w_names)
# plt.show(block=True)
