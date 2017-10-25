import numbers
import numpy as np
import collections


class Node:
    id = 0

    def __init__(self, children, name="Node"):
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name
        self.id = Node.id

        Node.id += 1

    def reverse_topo_sort(self):
        def topo_sort_dfs(node, visited, topo_sort):
            if node in visited:
                return topo_sort
            visited.add(node)
            for n in node.children:
                topo_sort = topo_sort_dfs(n, visited, topo_sort)
            return topo_sort + [node]

        return reversed(topo_sort_dfs(self, set(), []))

    def graph_name(self):
        return str(self.id)

    def __str__(self):
        return self.name  # + " " + str(self.id)

    def __add__(self, other):
        from automatic_differentiation.src.core.ops import Add
        return Add(self, other)

    def __neg__(self):
        from automatic_differentiation.src.core.ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from automatic_differentiation.src.core.ops import Mul
        return Mul(self, other)

    def __matmul__(self, other):
        from automatic_differentiation.src.core.ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from automatic_differentiation.src.core.ops import MatMul
        return MatMul(other, self)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        from automatic_differentiation.src.core.ops import Recipr
        return self.__mul__(Recipr(other))

    def __rtruediv__(self, other):
        from automatic_differentiation.src.core.ops import Recipr
        return Recipr(self).__mul__(other)

    def __pow__(self, power, modulo=None):
        from automatic_differentiation.src.core.ops import Pow
        return Pow(self, power)

    __rmul__ = __mul__
    __radd__ = __add__

    def __iter__(self):
        yield self
        for child in set(self.children):
            yield from child

    def add_node_subgraph_to_plot_graph(self, digraph):
        if self.graph_name() not in digraph.set_of_added_nodes:
            digraph.add_node_with_context(self, [])

            # Add connections to children, what if there's the same input twice to the operation?
            for child in self.children:
                digraph.add_edge(child, self)

            # Make each of the children do the same
            for child in self.children:
                child.add_node_subgraph_to_plot_graph(digraph)

    def __getitem__(self, item):
        from automatic_differentiation.src.core.reshape import Slice
        return Slice(self, item)

    def plot_comp_graph(self, view=True, name=None):
        from automatic_differentiation.src.visualization.graph_visualization import plot_comp_graph
        if name is None:
            name = "comp_graph"
        plot_comp_graph(self, view=view, name=name)


class Primitive(Node):
    epsilon = 1e-12

    def __init__(self, children, name=""):
        super().__init__(children, name)
        self.cached = None

    def __call__(self, *args, **kwargs):
        return self.eval()

    def eval(self):
        if self.cached is None:
            self.cached = self.f()

        return self.cached

    def f(self):
        raise NotImplementedError()

    def graph_df(self, wrt, curr_grad):
        raise NotImplementedError()


class Variable(Primitive):
    def __init__(self, value, name=None):
        if name is None:
            name = str(value)  # this op is really slow for np.arrays?!
        super().__init__([], name)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.cached = self._value = val

    def f(self):
        return self._value

    def graph_df(self, wrt, curr_grad):
        if self == wrt:
            return curr_grad
        return 0


def add_sum_name(node):
    return "'" + node.name + "' grad_sum"


def grad(top_node, wrt_list, curr_grad=Variable(1, name="init_grad")):
    assert isinstance(wrt_list, list) or isinstance(wrt_list, tuple)
    from automatic_differentiation.src.core.ops import Add

    dct = collections.defaultdict(list)
    dct[top_node].append(curr_grad)

    for node in top_node.reverse_topo_sort():
        dct[node] = Add(*dct[node], name=add_sum_name(node))

        for child in set(node.children):
            app = node.graph_df(wrt=child, curr_grad=dct[node])
            dct[child].append(app)

    return [dct[wrt] if isinstance(dct[wrt], Add) else Add(*dct[wrt], name=add_sum_name(wrt)) for wrt in wrt_list]
