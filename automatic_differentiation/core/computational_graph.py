import time
import numbers
import numpy as np
from contextlib import contextmanager


@contextmanager
def add_context(ctx):
    Node.context_list.append(ctx + "_" + str(time.time()))
    try:
        yield
    finally:
        del Node.context_list[-1]


class Node:
    id = 0
    context_list = []

    def __init__(self, children, name="Node"):
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name
        self.id = Node.id
        self.context_list = Node.context_list.copy()

        Node.id += 1

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
        from automatic_differentiation.src.core.high_level_ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from automatic_differentiation.src.core.high_level_ops import MatMul
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

    def __getitem__(self, item):
        from automatic_differentiation.src.core.reshape import Slice
        return Slice(self, item)

    def plot_comp_graph(self, view=True, name="comp_graph"):
        from automatic_differentiation.src.visualization import graph_visualization
        graph_visualization.plot_comp_graph(self, view=view, name=name)


class Primitive(Node):
    epsilon = 1e-12

    def __init__(self, children, name=""):
        super().__init__(children, name)
        self.cached = None
        self.shape = None

    def __call__(self, *args, **kwargs):
        return self.eval()

    def eval(self):
        if self.cached is None:
            self.cached = self._eval()

        return self.cached

    def _eval(self):
        raise NotImplementedError()

    def _partial_derivative(self, wrt, previous_grad):
        raise NotImplementedError()

    def partial_derivative(self, wrt, previous_grad):
        with add_context(self.name + "PD" + " wrt " + str(wrt)):
            return self._partial_derivative(wrt, previous_grad)


class Variable(Primitive):
    def __init__(self, value, name=None):
        if name is None:
            name = str(value)  # this op is really slow for np.arrays?!
        super().__init__([], name)

        if isinstance(value, numbers.Number):
            self._value = np.array(value, dtype=np.float64)
        else:
            self._value = value
        self.shape = self._value.shape

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.cached = self._value = val

    def _eval(self):
        return self._value

    def _partial_derivative(self, wrt, previous_grad):
        if self == wrt:
            return previous_grad
        return 0
