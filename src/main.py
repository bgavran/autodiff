from utils import *

"""
Plan:
* Idea to simplify comp. graphs might not be good, need to analyze why it's slow first!
* Profiling
* Grad buffer a good idea?
* 
"""

np.random.seed(1337)

x1 = Variable(name="x1")
w1 = Variable(name="w1")

graph = x1 @ w1

graph = Grad(graph, wrt=w1, expand_when_graphed=True)
plot_comp_graph(graph, view=False)

inpd = {x1: np.random.randn(2, 3), w1: np.random.randn(3, 5)}
res = graph.eval(inpd)
# print("x1:", x1.eval(inpd))
# print("----")
# print("w1:", w1.eval(inpd))
print("----")
print(res.shape)
